import os

# Configuration parameters using environment variables for security
CONFIG = {
    "MONGODB_URI": os.getenv("MONGODB_URI", "mongodb+srv://projects_db_user:QF1HNyiM29G6Lm5y@cluster0.3ghswog.mongodb.net/?appName=Cluster0"),
    "DB_NAME": os.getenv("DB_NAME", "proctoring_db"),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "violations"),
    "PORT": int(os.getenv("PORT", 8000)),
    "HOST": os.getenv("HOST", "0.0.0.0"),
    "ALLOWED_ORIGINS": os.getenv("ALLOWED_ORIGINS", "*").split(","),
    "AI_MODELS": {
        "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "object": "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite"
    }
}
