"""
Proctoring Environment - Python Backend
Detects: multiple faces, phone, face out of frame, head pose
Uses: OpenCV, face_recognition / dlib, YOLOv8 (ultralytics)
"""

import cv2
import numpy as np
import base64
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# ─── Load Models ────────────────────────────────────────────────
# Face detection via OpenCV DNN (no install needed beyond opencv-contrib)
FACE_PROTO = "deploy.prototext"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# We'll use OpenCV's built-in haar cascade as fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Try loading YOLOv8 for phone/object detection
yolo_available = False
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")   # auto-downloads on first run
    yolo_available = True
    print("[INFO] YOLOv8 loaded")
except ImportError:
    print("[WARN] ultralytics not installed — phone detection disabled")

# ─── Analysis ────────────────────────────────────────────────────

def decode_frame(b64: str) -> np.ndarray:
    img_data = base64.b64decode(b64.split(",")[-1])
    arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
    return faces


def estimate_head_pose(frame: np.ndarray, face_rect) -> str:
    """Simple gaze / head-turn estimate using eye positions."""
    x, y, w, h = face_rect
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,
                                        minNeighbors=5, minSize=(20, 20))
    if len(eyes) < 2:
        return "LOOKING_AWAY"
    # Sort eyes left-right
    eyes = sorted(eyes, key=lambda e: e[0])
    ex1, ey1, ew1, eh1 = eyes[0]
    ex2, ey2, ew2, eh2 = eyes[1]
    cx1 = ex1 + ew1 // 2
    cx2 = ex2 + ew2 // 2
    # If both eyes are visible and roughly symmetric → looking forward
    eye_spread = abs(cx2 - cx1)
    face_center_x = w // 2
    eyes_center = (cx1 + cx2) // 2
    offset = abs(eyes_center - face_center_x)
    if offset > face_center_x * 0.4:
        return "LOOKING_SIDEWAYS"
    return "LOOKING_FORWARD"


def detect_phone(frame: np.ndarray):
    """Detect cell phone using YOLO or fallback heuristic."""
    phones = []
    if yolo_available:
        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = yolo_model.names[cls_id]
            if label in ("cell phone", "remote", "tablet") and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phones.append({"label": label, "conf": round(conf, 2),
                                "bbox": [x1, y1, x2-x1, y2-y1]})
    return phones


def is_face_out_of_frame(frame: np.ndarray, faces) -> bool:
    if len(faces) == 0:
        return True
    h, w = frame.shape[:2]
    x, y, fw, fh = faces[0]
    # Face center must be within middle 60% of frame
    cx, cy = x + fw//2, y + fh//2
    margin_x, margin_y = w * 0.2, h * 0.2
    return not (margin_x < cx < w - margin_x and margin_y < cy < h - margin_y)


def analyze_frame(b64_frame: str) -> dict:
    frame = decode_frame(b64_frame)
    h, w = frame.shape[:2]

    faces      = detect_faces(frame)
    phones     = detect_phone(frame)
    face_count = len(faces)

    violations = []
    warnings   = []

    # ── Face checks ─────────────────────────────
    if face_count == 0:
        violations.append("NO_FACE_DETECTED")
    elif face_count > 1:
        violations.append(f"MULTIPLE_FACES_{face_count}")
    else:
        pose = estimate_head_pose(frame, faces[0])
        if pose == "LOOKING_AWAY":
            warnings.append("HEAD_TURNED_AWAY")
        elif pose == "LOOKING_SIDEWAYS":
            warnings.append("LOOKING_SIDEWAYS")

        if is_face_out_of_frame(frame, faces):
            violations.append("FACE_OUT_OF_FRAME")

    # ── Phone check ─────────────────────────────
    if phones:
        violations.append("PHONE_DETECTED")

    # ── Build annotated frame ────────────────────
    annotated = frame.copy()
    for (x, y, fw, fh) in faces:
        color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(annotated, "FACE", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for p in phones:
        bx, by, bw, bh = p["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
        cv2.putText(annotated, f"PHONE {p['conf']}", (bx, by-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Status overlay
    status_text = "✓ CLEAR" if not violations else "✗ VIOLATION"
    status_color = (0, 220, 0) if not violations else (0, 0, 255)
    cv2.putText(annotated, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return {
        "face_count":   face_count,
        "phones":       phones,
        "violations":   violations,
        "warnings":     warnings,
        "annotated":    annotated_b64,
        "timestamp":    time.time(),
        "yolo_enabled": yolo_available,
    }


# ─── Routes ──────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame provided"}), 400
    try:
        result = analyze_frame(data["frame"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "yolo": yolo_available})


if __name__ == "__main__":
    print("=" * 50)
    print("  Proctoring Server  — http://localhost:5050")
    print(f"  YOLO phone detect : {'ENABLED' if yolo_available else 'DISABLED (pip install ultralytics)'}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
