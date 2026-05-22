# ProctorKit Documentation

## 1. Overview
ProctorKit is a modular, production-grade SDK for browser-based AI proctoring.

## 2. Integration
```javascript
import { ProctorKit } from './sdk/ProctorKit.js';

const proctor = new ProctorKit({
    apiBase: 'http://localhost:8000',
    examSessionId: 'SESSION_ID',
    features: {
        face: true,
        phone: true,
        gaze: true
    },
    onDetection: (data) => console.log(data),
    onViolation: (data) => alert("Violation: " + data.msg)
});

await proctor.init();
proctor.start(videoElement, canvasElement);
```

## 3. Configuration
- `features`: Object to toggle detection modules.
  - `face`: Enables face landmarker.
  - `phone`: Enables object detector.
  - `gaze`: Enables gaze estimation.
- `detectionInterval`: Throttling control.
