# ProctorKit: Technical Architecture Deep Dive

## 1. Executive Summary
ProctorKit is designed as a **Hybrid Inference Engine**. It distributes the computational load of AI proctoring between the user's browser (Edge Computing) and a centralized server (Cloud Validation). This approach ensures sub-second latency for user feedback while maintaining the high security and auditability required for high-stakes assessments.

## 2. Component Diagram

### 2.1 The Edge (Client-Side SDK)
The SDK is the primary data collector. It operates on a "Trust but Verify" principle.
- **Inference Engine:** MediaPipe (WASM).
- **Responsibility:** 
    - Real-time face tracking and object detection.
    - Temporal analysis (filtering noise/flicker).
    - Local state management (violation counts, session time).
    - Evidence capture (Screenshots).

### 2.2 The Nexus (Backend API)
The Backend is the source of truth and the secondary validator.
- **Framework:** FastAPI (Asynchronous).
- **Responsibility:**
    - High-integrity violation reporting.
    - Secondary validation using YOLOv8 (Server-side inference).
    - Multi-tenant session management.

### 2.3 The Vault (Data Layer)
- **Primary Storage:** MongoDB. Chosen for its flexibility in storing rich, unstructured metadata accompanying violation reports.
- **Cache Layer:** Redis (Planned for production). Used for session heartbeat monitoring and rate-limiting.

## 3. Communication Protocols
- **API (REST):** Used for initial session handshake and violation reporting.
- **WebSockets (Proposed for Rebuild):** Real-time "Live Proctoring" dashboard updates and remote session termination commands.

## 4. Security Model
- **Client-Side:** Integrity checks to ensure the SDK hasn't been tampered with (e.g., threshold modifications).
- **Server-Side:** Cross-referencing client reports with server-side YOLO inference to detect "Blind Spots" or malicious SDK bypasses.

## 5. Deployment Architecture
- **Containerization:** Dockerized microservices.
- **Orchestration:** Kubernetes for horizontal scaling of the API workers based on the number of active exam sessions.
