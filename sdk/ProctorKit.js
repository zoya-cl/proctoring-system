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
        const now = Date.now();
        const intervalMs = 5000; // run checks every 5 seconds for better detection accuracy

        if (now - this.lastDetectionTs < intervalMs) {
            requestAnimationFrame(() => this.loop());
            return;
        }

        this.lastDetectionTs = now;
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

        const phoneDetections = rawPhones.filter((d) => {
            const label = d.categories?.[0]?.categoryName || "";
            return /phone|cell|mobile/i.test(label);
        });

        // Temporal buffer for phone detection to reduce false positives
        if (phoneDetections.length > 0) this.phoneFrames++;
        else this.phoneFrames = 0;
        const isPhoneDetected = this.phoneFrames > 3; // Requires several checks in a row
        
        const isGazeOnScreen = faces.length > 0 && Math.abs(landmarks[1].x - ((landmarks[33].x + landmarks[263].x) / 2)) < 0.12;
        const data = {
            faces: faces.length,
            phones: isPhoneDetected ? phoneDetections.length : 0,
            elapsed: Math.floor((Date.now() - this.startTime) / 1000) + "s",
            violations: this.violCount || 0,
            checks: {
                face: faces.length > 0,
                phone: !isPhoneDetected,
                multi: faces.length === 1,
                gaze: isGazeOnScreen,
                frame: faces.length > 0
            }
        };

        if (this.onResult) this.onResult(data);

        const isNoFace = faces.length === 0;
        const isMultiFace = faces.length > 1;
        const isPhone = isPhoneDetected;

        if (isNoFace) {
            this.consecutive.noFace += 1;
        } else {
            this.consecutive.noFace = 0;
        }
        if (isMultiFace) {
            this.consecutive.multiFace += 1;
        } else {
            this.consecutive.multiFace = 0;
        }
        if (isPhone) {
            this.consecutive.phone += 1;
        } else {
            this.consecutive.phone = 0;
        }

        const now = Date.now();
        const minViolationInterval = 20_000; // 20 seconds
        const shouldReport = (
            (isNoFace && this.consecutive.noFace >= 2) ||
            (isMultiFace && this.consecutive.multiFace >= 2) ||
            (isPhone && this.consecutive.phone >= 2)
        ) && now - this.lastViolationTs >= minViolationInterval;

        if (shouldReport) {
            this.violCount++;
            this.lastViolationTs = now;
            const msg = isNoFace ? "No face" : isPhone ? "Phone detected" : "Multiple faces";
            this.report(msg, data);
            if(this.config.onViolation) this.config.onViolation({ msg, type: msg, faces: faces.length, phones: phoneDetections.length, checks: data.checks, elapsed: data.elapsed });
        }
    }

    async report(message, meta) {
        // Send directly to NestJS backend, bypassing FastAPI layer
        const payload = { 
            userId: this.config.userId,
            interviewId: this.config.interviewId,
            message: message, 
            screenshot: "data:image/jpeg;base64,...",
            testType: "interview",
            meta: meta,
            timestamp: new Date().toISOString()
        };
        console.log("📤 Sending violation to NestJS:", payload);
        try {
            const nestjsBackend = this.config.nestjsBackend || "http://localhost:8080";
            const response = await fetch(`${nestjsBackend}/ai-interview/report-violation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            console.log("✅ Violation reported successfully:", result);
        } catch (e) { console.error("❌ ProctorKit Report Error:", e); }
    }
}
