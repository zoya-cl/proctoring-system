from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from models.database import db
from config.settings import CONFIG
from datetime import datetime
import cv2
import numpy as np
import base64

# Attempt to load YOLO locally on server for validation
try:
    from ultralytics import YOLO
    server_model = YOLO("yolov8n.pt") 
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

app = FastAPI(title="Proctoring Microservice - Integrated with NestJS")
app.add_middleware(CORSMiddleware, allow_origins=CONFIG["ALLOWED_ORIGINS"], allow_methods=["*"], allow_headers=["*"])

import urllib.request
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)

def forward_violation_to_nestjs(report_data: dict):
    nestjs_url = CONFIG.get("NESTJS_BACKEND", "http://localhost:8080").rstrip("/")
    endpoint = f"{nestjs_url}/ai-interview/report-violation"
    
    violation_type = (
        report_data.get("meta", {}).get("type")
        or report_data.get("type")
        or "Unknown"
    )
    interview_id = report_data.get("interviewId") or report_data.get("session_id", "")
    user_id = report_data.get("userId", "")

    payload = {
        "userId":      user_id,
        "interviewId": interview_id,
        "type":        violation_type,
        "message":     report_data.get("message", "Violation detected"),
        "screenshot":  report_data.get("screenshot", ""),
        "meta":        report_data.get("meta", {}),
    }

    print(f"\n🔁 [Proctor] Forwarding violation → {endpoint}")
    print(f"   interviewId : {interview_id}")
    print(f"   userId      : {user_id}")
    print(f"   type        : {violation_type}")
    print(f"   message     : {payload['message']}")
    print(f"   screenshot  : {'yes' if payload['screenshot'] else 'no'}")

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            print(f"✅ [Proctor] Violation forwarded successfully → {res_body[:200]}")
            return json.loads(res_body)
    except Exception as e:
        print(f"❌ [Proctor] Failed to forward violation to NestJS ({endpoint}): {e}")
        return None

async def async_forward_violation(report_data: dict):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, forward_violation_to_nestjs, report_data)

from typing import Optional

# Updated schema to include user/interview tracking
class ViolationReport(BaseModel):
    session_id: str
    message: str
    meta: dict = {}
    screenshot: str
    userId: str = Field(..., description="User ID from NestJS backend")
    interviewId: str = Field(None, description="Interview ID if applicable")
    testType: str = Field("interview", description="Type: coding, interview, or test")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FrameValidation(BaseModel):
    session_id: str
    frame: str # Base64
    userId: str = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ProctorKit Microservice",
        "nestjs_integrated": True
    }

@app.post("/report")
async def report_violation(report: ViolationReport):
    """
    Report a proctoring violation with user/interview context
    
    - userId: Required - User ID from NestJS backend
    - interviewId: Interview/test ID
    - testType: 'coding', 'interview', or 'test'
    """
    try:
        violation_data = report.dict()
        # Ensure timestamp is set
        if "timestamp" not in violation_data:
            violation_data["timestamp"] = datetime.utcnow()
        
        result = await db.save_violation(violation_data)
        
        # Asynchronously forward the violation report to NestJS in the background
        asyncio.create_task(async_forward_violation(violation_data))
        
        return {
            "status": "success",
            "violation_id": str(result.inserted_id),
            "userId": report.userId,
            "interviewId": report.interviewId,
            "testType": report.testType
        }
    except Exception as e:
        print(f"Error handling report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai-interview/report-violation")
async def report_violation_nestjs(report: ViolationReport):
    """
    NestJS-compatible violation endpoint (mirrors /report)
    
    - userId: Required - User ID from NestJS backend
    - interviewId: Interview/test ID
    - testType: 'coding', 'interview', or 'test'
    """
    try:
        violation_data = report.dict()
        # Ensure timestamp is set
        if "timestamp" not in violation_data:
            violation_data["timestamp"] = datetime.utcnow()
        
        result = await db.save_violation(violation_data)
        
        # Asynchronously forward the violation report to NestJS in the background
        asyncio.create_task(async_forward_violation(violation_data))
        
        return {
            "data": {
                "violation_id": str(result.inserted_id),
                "userId": report.userId,
                "interviewId": report.interviewId,
                "testType": report.testType
            }
        }
    except Exception as e:
        print(f"Error handling report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/violations/{user_id}")
async def get_user_violations(user_id: str):
    """Get all violations for a specific user"""
    try:
        violations = await db.get_violations_by_user(user_id)
        return {
            "userId": user_id,
            "total_violations": len(violations),
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interview/{interview_id}")
async def get_interview_violations(interview_id: str):
    """Get all violations for a specific interview/test"""
    try:
        violations = await db.get_violations_by_interview(interview_id)
        return {
            "interviewId": interview_id,
            "total_violations": len(violations),
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-frame")
async def validate_frame(data: FrameValidation):
    """
    Optional Server-Side processing: Allows the server to verify the client's report.
    """
    if not MODEL_LOADED:
        return {"status": "error", "detail": "Server-side inference model not loaded"}

    try:
        # Decode frame
        img_data = base64.b64decode(data.frame.split(",")[-1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference
        results = server_model(frame, verbose=False)[0]
        phones = [box for box in results.boxes if int(box.cls[0]) in [67, 73, 74]] # Phone/Remote/Tablet classes
        
        return {
            "status": "validated",
            "phone_detected": len(phones) > 0,
            "phone_count": len(phones)
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=CONFIG["HOST"], port=CONFIG["PORT"], reload=True)
