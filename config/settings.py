import os

# Configuration parameters using environment variables for security
CONFIG = {
    # Fulcrum Staging MongoDB (integrated with backend-nestjs)
    "MONGODB_URI": os.getenv("MONGODB_URI", "mongodb+srv://fulcrum-staging:7HTX4GmSsWaDwKVr@cluster0.h67ykec.mongodb.net/fulcrum-staging"),
    "DB_NAME": os.getenv("DB_NAME", "fulcrum-staging"),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "proctoringviolations"),
    "PORT": int(os.getenv("PORT", 8000)),
    "HOST": os.getenv("HOST", "0.0.0.0"),  # Allow connections from NestJS
    "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1,http://localhost:8000,http://localhost:3000,http://localhost:3001").split(","),
    "AI_MODELS": {
        "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "object": "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite"
    },
    # Integration settings
    "NESTJS_BACKEND": os.getenv("NESTJS_BACKEND", "http://localhost:8080"),
    "TRACK_USER_SESSIONS": True,  # Track user + interview data
}
