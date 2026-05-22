import { FaceLandmarker, ObjectDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

/**
 * ProctorKit SDK - Robust Data Pipeline
 */
export class ProctorKit {
    constructor(config) {
        this.config = config;
        this.active = false;
        this.startTime = null;
        this.violCount = 0;
        this.phoneFrames = 0; // Buffer for phone detection
        this.runners = { face: null, object: null };
        this.onDraw = null;
        this.onResult = null;
    }

    async init(features = { face: true, phone: true }) {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        if (features.face) {
            this.runners.face = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
                runningMode: "VIDEO", numFaces: 1
            });
        }
        if (features.phone) {
            this.runners.object = await ObjectDetector.createFromOptions(vision, {
                baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite" },
                runningMode: "VIDEO"
            });
        }
    }

    start(videoEl, onResult) {
        this.video = videoEl;
        this.onResult = onResult;
        this.active = true;
        this.startTime = Date.now();
        this.loop();
    }

    loop() {
        if (!this.active) return;
        if (this.video.readyState >= 2) {
            const ts = performance.now();
            const res = {
                face: this.runners.face ? this.runners.face.detectForVideo(this.video, ts) : null,
                phone: this.runners.object ? this.runners.object.detectForVideo(this.video, ts) : null
            };
            this.process(res);
        }
        requestAnimationFrame(() => this.loop());
    }

    process(res) {
        const faces = res.face?.faceLandmarks || [];
        const rawPhones = res.phone?.detections || [];
        const landmarks = faces.length > 0 ? faces[0] : null;

        // Temporal buffer for phone detection to reduce false positives
        if (rawPhones.length > 0) this.phoneFrames++;
        else this.phoneFrames = 0;
        const isPhoneDetected = this.phoneFrames > 5; // Requires 5 consecutive frames
        
        const data = {
            faces: faces.length,
            phones: isPhoneDetected ? rawPhones.length : 0,
            elapsed: Math.floor((Date.now() - this.startTime) / 1000) + "s",
            violations: this.violCount || 0,
            checks: {
                face: faces.length > 0,
                phone: !isPhoneDetected,
                multi: faces.length === 1,
                gaze: faces.length > 0 && Math.abs(landmarks[1].x - (landmarks[33].x + landmarks[263].x)/2) < 0.08,
                frame: faces.length > 0
            }
        };

        if (this.onResult) this.onResult(data);

        if (faces.length === 0 || faces.length > 1 || isPhoneDetected) {
            this.violCount++;
            const msg = faces.length === 0 ? "No face" : isPhoneDetected ? "Phone detected" : "Multiple faces";
            this.report(msg, data);
            if(this.config.onViolation) this.config.onViolation({ msg });
        }
    }

    async report(message, meta) {
        const payload = { 
            session_id: this.config.examSessionId, 
            message: message, 
            meta: meta,
            screenshot: "data:image/jpeg;base64,..." // Placeholder since evidence capture is handled by caller
        };
        console.log("Reporting Payload:", payload);
        try {
            await fetch(`${this.config.apiBase}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        } catch (e) { console.error("ProctorKit Report Error:", e); }
    }
}
