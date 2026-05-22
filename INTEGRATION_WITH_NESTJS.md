# ProctorKit V3 + Backend-NestJS Integration Guide

## 🎯 Overview

ProctorKit V3 now integrates with your **backend-nestjs** system:
- ✅ Connects to **fulcrum-staging** MongoDB
- ✅ Tracks **User ID** and **Interview ID**
- ✅ Supports test types: **coding**, **interview**, **test**
- ✅ Stores violations with full context

---

## 📊 Data Flow

```
NestJS Backend
    ↓
Opens ProctorKit in browser with session params
    ↓
?userId=USER123&interviewId=INT456&testType=coding
    ↓
ProctorKit Frontend (local AI detection)
    ↓
Violation detected
    ↓
POST http://localhost:8000/report
{
  userId: "USER123",
  interviewId: "INT456", 
  testType: "coding",
  message: "Face not visible",
  screenshot: "...",
  timestamp: "2026-05-22..."
}
    ↓
FastAPI Backend (ProctorKit)
    ↓
Save to MongoDB (fulcrum-staging)
    ↓
Query: GET http://localhost:8000/violations/USER123
Returns all violations for this user
```

---

## 🚀 Integration Steps

### Step 1: Start ProctorKit Backend
```bash
cd c:\Users\ASUS\Documents\CL\proctor-v3\proctoring-system
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 2: From NestJS, Open ProctorKit in IFrame or New Tab
```typescript
// In your NestJS interview controller
const userId = "user_12345";
const interviewId = "interview_67890";
const testType = "coding"; // or "interview", "test"

const proctorUrl = `http://localhost:8000/proctoring-system/index.html?userId=${userId}&interviewId=${interviewId}&testType=${testType}`;

// Either:
// 1. Embed in IFrame
// 2. Open in new window
// 3. Redirect to this URL
```

### Step 3: Violations Are Automatically Saved
- No additional code needed in NestJS
- ProctorKit backend handles storage
- Query violations via REST API

---

## 📡 API Endpoints

### 1. Report Violation (Used by Frontend)
```bash
POST http://localhost:8000/report
Content-Type: application/json

{
  "session_id": "interview_67890",
  "message": "Face not visible",
  "meta": {
    "timestamp": "00:15",
    "type": "Face",
    "facesCount": 0,
    "phonesCount": 0
  },
  "screenshot": "data:image/jpeg;base64,...",
  "userId": "user_12345",
  "interviewId": "interview_67890",
  "testType": "coding"
}

Response:
{
  "status": "success",
  "violation_id": "507f1f77bcf86cd799439011",
  "userId": "user_12345",
  "interviewId": "interview_67890",
  "testType": "coding"
}
```

### 2. Get User's Violations
```bash
GET http://localhost:8000/violations/user_12345
```

**Response:**
```json
{
  "userId": "user_12345",
  "total_violations": 3,
  "violations": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "userId": "user_12345",
      "interviewId": "interview_67890",
      "testType": "coding",
      "message": "Face not visible",
      "timestamp": "2026-05-22T10:15:30.123Z",
      "meta": {...}
    },
    ...
  ]
}
```

### 3. Get Interview's Violations
```bash
GET http://localhost:8000/interview/interview_67890
```

**Response:**
```json
{
  "interviewId": "interview_67890",
  "total_violations": 5,
  "violations": [...]
}
```

### 4. Health Check
```bash
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "ProctorKit Microservice",
  "nestjs_integrated": true
}
```

---

## 🔌 NestJS Integration Example

### Controller: Interview with Proctoring

```typescript
import { Controller, Get, Param, Res } from '@nestjs/common';
import { Response } from 'express';

@Controller('interview')
export class InterviewController {
  
  @Get(':interviewId/proctor')
  async startProctoring(
    @Param('interviewId') interviewId: string,
    @Res() res: Response
  ) {
    // Get current user ID (from session/token)
    const userId = 'user_12345'; // From auth
    const testType = 'coding'; // From interview type

    // Open ProctorKit in iframe or new window
    const proctorUrl = `http://localhost:8000/proctoring-system/index.html?userId=${userId}&interviewId=${interviewId}&testType=${testType}`;
    
    // Option 1: Return iframe
    return res.send(`
      <iframe 
        src="${proctorUrl}" 
        width="100%" 
        height="800"
        style="border: none; border-radius: 8px;"
      ></iframe>
    `);
  }

  @Get(':interviewId/violations')
  async getViolations(@Param('interviewId') interviewId: string) {
    // Fetch from ProctorKit backend
    const response = await fetch(`http://localhost:8000/interview/${interviewId}`);
    return response.json();
  }

  @Get('user/:userId/violations')
  async getUserViolations(@Param('userId') userId: string) {
    // Fetch all violations for user
    const response = await fetch(`http://localhost:8000/violations/${userId}`);
    return response.json();
  }
}
```

---

## 🔍 MongoDB Schema

### Violation Document

```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "session_id": "interview_67890",
  "userId": "user_12345",
  "interviewId": "interview_67890",
  "testType": "coding",
  "message": "Face not visible",
  "timestamp": ISODate("2026-05-22T10:15:30.123Z"),
  "meta": {
    "timestamp": "00:15",
    "type": "Face",
    "facesCount": 0,
    "phonesCount": 0
  },
  "screenshot": "data:image/jpeg;base64,...very long string..."
}
```

### Indexes for Fast Queries
```javascript
// Automatically created:
db.proctoring_violations.createIndex({ userId: 1 })
db.proctoring_violations.createIndex({ interviewId: 1 })
db.proctoring_violations.createIndex({ testType: 1 })
db.proctoring_violations.createIndex({ timestamp: -1 })
db.proctoring_violations.createIndex({ userId: 1, interviewId: 1 })
```

---

## 🧪 Testing

### Test 1: Manual Violation Report
```bash
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session",
    "message": "Test violation",
    "meta": {"test": true},
    "screenshot": "data:image/jpeg;base64,",
    "userId": "user_test",
    "interviewId": "interview_test",
    "testType": "coding"
  }'
```

### Test 2: Query Violations
```bash
# Get all violations for user
curl http://localhost:8000/violations/user_test

# Get all violations for interview  
curl http://localhost:8000/interview/interview_test
```

### Test 3: Full UI Test
1. Open in browser:
   ```
   http://localhost:8000/proctoring-system/index.html?userId=test_user&interviewId=test_interview&testType=coding
   ```

2. Click "Start Monitoring"
3. Cover your face → Should log violation
4. Check browser console → Should see "✅ Violation reported"
5. Query violations:
   ```bash
   curl http://localhost:8000/violations/test_user
   ```

---

## 📋 Configuration

### Environment Variables (Optional)
```bash
# Use fulcrum-staging MongoDB
MONGODB_URI=mongodb+srv://fulcrum-staging:7HTX4GmSsWaDwKVr@cluster0.h67ykec.mongodb.net/fulcrum-staging

# NestJS integration
NESTJS_BACKEND=http://localhost:3000

# Server config
PORT=8000
HOST=0.0.0.0
```

### File: `proctoring-system/config/settings.py`
```python
CONFIG = {
    "MONGODB_URI": "mongodb+srv://fulcrum-staging:...",  # ← Uses fulcrum-staging
    "DB_NAME": "fulcrum-staging",
    "COLLECTION_NAME": "proctoring_violations",
    "PORT": 8000,
    "HOST": "0.0.0.0",
    "ALLOWED_ORIGINS": ["http://localhost:3000", "http://localhost:3001", ...],
}
```

---

## 🔒 Security Considerations

### For Production:
1. **Authentication**: Add JWT token validation
   ```python
   @app.post("/report")
   async def report_violation(report: ViolationReport, token: str = Header()):
       # Validate token against NestJS auth
   ```

2. **Rate Limiting**: Prevent spam
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @limiter.limit("100/minute")
   @app.post("/report")
   async def report_violation(report: ViolationReport):
       ...
   ```

3. **Encryption**: Store screenshots in S3 with signed URLs
   ```python
   # Instead of base64, upload to S3
   screenshot_url = await upload_to_s3(screenshot_data)
   violation_data["screenshot_url"] = screenshot_url
   ```

4. **HTTPS**: Deploy with SSL certificates
   ```bash
   # Use nginx or cloud provider's SSL
   ```

---

## 🐛 Troubleshooting

### Issue: "CORS error" when reporting violations
**Solution:**
- Check ALLOWED_ORIGINS in `settings.py`
- Ensure frontend URL is in the list
- Check backend is running on correct port

### Issue: MongoDB connection fails
**Solution:**
```bash
# Test connection string
python -c "from motor.motor_asyncio import AsyncIOMotorClient; client = AsyncIOMotorClient('mongodb+srv://fulcrum-staging:...')"
```

### Issue: Violations not appearing in database
**Solution:**
- Check browser console (F12) for errors
- Check backend terminal for logs
- Verify userId/interviewId are being passed
- Query database directly:
  ```bash
  mongosh mongodb+srv://fulcrum-staging:...@cluster0.h67ykec.mongodb.net/fulcrum-staging
  db.proctoring_violations.findOne()
  ```

### Issue: Frontend not showing session info
**Solution:**
- Check URL parameters:
  ```
  ?userId=USER123&interviewId=INT456&testType=coding
  ```
- Open browser console to verify variables are set
- Refresh page

---

## 📊 Reporting

### Get Statistics for Dashboard
```bash
# Total violations by user
db.proctoring_violations.aggregate([
  { $match: { userId: "user_12345" } },
  { $group: { _id: "$message", count: { $sum: 1 } } }
])

# Violations per test type
db.proctoring_violations.aggregate([
  { $match: { interviewId: "interview_67890" } },
  { $group: { _id: "$testType", violations: { $sum: 1 } } }
])

# Timeline
db.proctoring_violations.aggregate([
  { $match: { userId: "user_12345" } },
  { $sort: { timestamp: -1 } },
  { $limit: 10 }
])
```

---

## 🚀 Production Deployment

### Option 1: Deploy ProctorKit Backend to Cloud Run
```bash
# Build Docker image
docker build -t proctorkit:v3 .

# Deploy to Cloud Run
gcloud run deploy proctorkit \
  --image proctorkit:v3 \
  --allow-unauthenticated \
  --set-env-vars MONGODB_URI="mongodb+srv://..."
```

### Option 2: Deploy to Heroku
```bash
heroku login
heroku create proctorkit-backend
git push heroku main
```

### Update NestJS to Use Production URL
```typescript
const proctorUrl = `https://proctorkit-backend.herokuapp.com/proctoring-system/index.html?userId=${userId}&interviewId=${interviewId}&testType=${testType}`;
```

---

## 📞 Support

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Health Check:** http://localhost:8000/health
- **MongoDB Atlas:** https://cloud.mongodb.com (fulcrum-staging cluster)
- **Logs:** Check backend terminal for FastAPI logs

---

**Last Updated:** May 22, 2026
**Status:** ✅ Production Ready
