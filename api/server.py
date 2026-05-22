from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.database import db
from config.settings import CONFIG
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

app = FastAPI(title="Proctoring Microservice")
app.add_middleware(CORSMiddleware, allow_origins=CONFIG["ALLOWED_ORIGINS"], allow_methods=["*"], allow_headers=["*"])

class ViolationReport(BaseModel):
    session_id: str
    message: str
    meta: dict
    screenshot: str

class FrameValidation(BaseModel):
    session_id: str
    frame: str # Base64

@app.post("/report")
async def report_violation(report: ViolationReport):
    try:
        await db.save_violation(report.dict())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-frame")
async def validate_frame(data: FrameValidation):
    """
    Optional Server-Side processing: Allows the server to verify the client's report.
    """
    if not MODEL_LOADED:
        return {"status": "error", "detail": "Server-side inference model not loaded"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=CONFIG["HOST"], port=CONFIG["PORT"], reload=True)
