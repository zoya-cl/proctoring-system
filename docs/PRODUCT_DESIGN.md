# ProctorKit: Product Design & Software Requirements Specification

## 1. Introduction
ProctorKit is a modular, high-performance AI proctoring solution designed for modern web-based assessments. It provides a hybrid detection architecture that balances low-latency client-side monitoring with secure, server-side validation and audit persistence.

## 2. Software Requirements Specification (SRS)

### 2.1 Functional Requirements
- **Real-time Monitoring:** The system must detect faces and objects (e.g., mobile phones) in real-time within the user's browser.
- **Violation Detection:**
    - **Face Presence:** Detect if the user is missing from the frame.
    - **Multiple Faces:** Detect if more than one person is present in the frame.
    - **Object Detection:** Detect unauthorized objects like mobile phones.
    - **Gaze Estimation:** (Experimental) Detect if the user is looking away from the screen.
- **Evidence Capture:** The system must capture metadata and screenshots when a violation is detected.
- **Violation Reporting:** The SDK must report violations to a central server for audit logging.
- **Server-Side Validation:** The server must provide an endpoint to re-verify frames using a more robust model (YOLOv8) to confirm or refute client-side reports.
- **Data Persistence:** All violation records must be stored in a persistent database for later review.

### 2.2 Non-Functional Requirements
- **Low Latency:** Client-side detection must run at a minimum of 15 FPS to provide immediate feedback.
- **Scalability:** The backend must handle concurrent violation reports from multiple examination sessions.
- **Accuracy:** Minimize false positives through temporal buffering (e.g., requiring multiple consecutive frames of detection before flagging a phone).
- **Security:** Implement server-side validation to prevent tampering with client-side detection logic.
- **Cross-Browser Compatibility:** The SDK must work on all modern browsers supporting WebAssembly (WASM).

## 3. Product Design Document (PDD)

### 3.1 System Architecture
ProctorKit follows a **Hybrid Detection Architecture**:
1.  **Client-Side (SDK):** Uses MediaPipe and WASM for on-device inference.
2.  **Server-Side (API):** Built with FastAPI for high-performance reporting and secondary YOLOv8 validation.
3.  **Persistence Layer:** MongoDB (via Motor) for storing unstructured violation data and metadata.

### 3.2 Component Design

#### 3.2.1 ProctorKit SDK (JavaScript)
- **Engine:** MediaPipe Tasks Vision.
- **Models:**
    - Face Landmarker (Face detection and gaze estimation).
    - Object Detector (SSD MobileNet V2 for phone detection).
- **Logic:**
    - `loop()`: Continuous frame processing using `requestAnimationFrame`.
    - `process()`: Analyzes inference results, applies temporal buffering (e.g., 5 consecutive frames for phone detection), and triggers reporting.
    - `report()`: Asynchronous transmission of violation data to the backend.

#### 3.2.2 Backend API (Python/FastAPI)
- **Endpoints:**
    - `POST /report`: Saves violation metadata to MongoDB.
    - `POST /validate-frame`: Decodes base64 frames and runs YOLOv8n inference for high-accuracy spot-checks.
- **Database Integration:** Uses `motor.motor_asyncio` for non-blocking I/O with MongoDB.

#### 3.2.3 Data Schema (MongoDB)
Violations are stored with the following structure:
```json
{
  "session_id": "string",
  "message": "string (e.g., 'Phone detected')",
  "meta": {
    "faces": "number",
    "phones": "number",
    "elapsed": "string",
    "checks": { "face": "bool", "phone": "bool", ... }
  },
  "screenshot": "string (base64 image data)"
}
```

### 3.3 Demo Application (Frontend)
The project includes a production-grade demo (`index.html`) featuring:
- **Telemetry Stats:** Real-time display of face count, total violations, and session duration.
- **Incident Log:** A chronological feed of detected violations with timestamps.
- **Real-time Analysis Badges:** Visual indicators (Face Presence, Single Identity, Gaze, Object Detection) that change color based on the current detection state.
- **Visual Overlays:** Canvas-based drawing of face mesh and object bounding boxes over the live video stream.

### 3.4 Data Flow
1.  **Initialization:** SDK loads WASM binaries and AI models.
2.  **Inference:** SDK processes video frames locally.
3.  **Detection:** If a violation threshold is met, the SDK captures a snapshot.
4.  **Reporting:** SDK sends the report to the `/report` endpoint.
5.  **Audit (Optional):** The frontend or proctor dashboard calls `/validate-frame` for server-side confirmation of the violation.
6.  **Persistence:** Backend saves the record to MongoDB.

## 4. Production Considerations (The "Blueprint" for Rebuild)

### 4.1 Security & Authentication
- **Authentication:** All API endpoints must be protected using JWT (JSON Web Tokens). The SDK must include a `token` in the header of all requests.
- **HTTPS/TLS:** All communication between the SDK and Server must be over encrypted channels.
- **Data Privacy:** Personal Identifiable Information (PII) should be handled according to GDPR/CCPA standards. Screenshots should be stored in secure, private S3 buckets with signed URL access.

### 4.2 Scalability & Availability
- **Load Balancing:** Use a load balancer (e.g., Nginx or AWS ALB) to distribute traffic across multiple FastAPI workers.
- **Inference Throttling:** Implement a queue system (e.g., RabbitMQ or Redis) for server-side frame validation to prevent overloading during peak exam times.
- **Database Indexing:** Ensure MongoDB has proper indexes on `session_id` and `timestamp` for fast retrieval of audit logs.

### 4.3 Reliability & Error Handling
- **Graceful Degradation:** If the backend is unreachable, the SDK should cache violation reports locally (in IndexedDB) and retry synchronization once the connection is restored.
- **Centralized Logging:** Implement structured logging (e.g., ELK stack or Sentry) to monitor for SDK crashes and API performance bottlenecks.

### 4.4 API Versioning
- Use URI versioning (e.g., `/api/v1/...`) to ensure backward compatibility as the detection logic and reporting schema evolve.

## 5. Testing Strategy
- **Unit Testing:**
    - **SDK:** Test the `process()` logic with mock inference results.
    - **Backend:** Test Pydantic models and database helper functions using `pytest`.
- **Integration Testing:** Verify the end-to-end flow from detection in the browser to persistence in MongoDB.
- **Performance Testing:** Simulate 100+ concurrent proctoring sessions to measure server latency and database write performance.

## 6. Technology Stack
- **Frontend:** TypeScript (for type safety), MediaPipe WASM, React/Angular (for componentization).
- **Backend:** Python 3.9+, FastAPI, Ultralytics YOLOv8, Celery (for async tasks).
- **Database:** MongoDB (Audit logs), Redis (Session state & Caching).
- **Infrastructure:** Docker, Kubernetes, Prometheus/Grafana (Monitoring).
