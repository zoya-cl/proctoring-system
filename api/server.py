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

# Updated schema to include user/interview tracking
class ViolationReport(BaseModel):
    session_id: str
    message: str
    meta: dict
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
        
        return {
            "status": "success",
            "violation_id": str(result.inserted_id),
            "userId": report.userId,
            "interviewId": report.interviewId,
            "testType": report.testType
        }
    except Exception as e:
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
