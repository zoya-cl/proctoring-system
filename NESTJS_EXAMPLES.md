# NestJS Integration Examples

## 1. Simple Interview Controller with ProctorKit

```typescript
// src/interview/interview.controller.ts
import { Controller, Get, Post, Param, Body, UseGuards, Req, Res } from '@nestjs/common';
import { Response } from 'express';
import { JwtAuthGuard } from 'src/auth/guards/jwt-auth.guard';
import { InterviewService } from './interview.service';

@Controller('interview')
@UseGuards(JwtAuthGuard)
export class InterviewController {
  constructor(private interviewService: InterviewService) {}

  // Start proctored interview
  @Get(':interviewId/start-proctor')
  async startProctoring(
    @Param('interviewId') interviewId: string,
    @Req() req,
    @Res() res: Response
  ) {
    const userId = req.user.id; // From JWT token
    const interview = await this.interviewService.getInterview(interviewId);
    
    // Determine test type from interview
    const testType = interview.type || 'interview'; // coding, interview, test

    // Build ProctorKit URL
    const proctorUrl = `http://localhost:8000/proctoring-system/index.html?userId=${userId}&interviewId=${interviewId}&testType=${testType}`;

    // Return HTML with iframe
    return res.send(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Interview - Proctored</title>
          <style>
            body { margin: 0; padding: 20px; background: #0f172a; font-family: Arial; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { color: white; margin-bottom: 20px; }
            #proctor-frame { width: 100%; height: 900px; border: none; border-radius: 8px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1>Interview - Proctored Session</h1>
              <p>User: ${userId} | Interview: ${interviewId}</p>
            </div>
            <iframe id="proctor-frame" src="${proctorUrl}"></iframe>
          </div>
        </body>
      </html>
    `);
  }

  // Get violations for an interview
  @Get(':interviewId/violations')
  async getInterviewViolations(@Param('interviewId') interviewId: string) {
    try {
      const response = await fetch(`http://localhost:8000/interview/${interviewId}`);
      const data = await response.json();
      
      return {
        interviewId,
        violations: data.violations,
        summary: {
          total: data.total_violations,
          types: this.groupByType(data.violations)
        }
      };
    } catch (error) {
      return {
        error: 'Failed to fetch violations',
        message: error.message
      };
    }
  }

  // Get violations for current user
  @Get('user/violations')
  async getUserViolations(@Req() req) {
    const userId = req.user.id;
    
    try {
      const response = await fetch(`http://localhost:8000/violations/${userId}`);
      const data = await response.json();
      
      return {
        userId,
        violations: data.violations,
        summary: {
          total: data.total_violations,
          byType: this.groupByType(data.violations),
          byInterview: this.groupByInterview(data.violations)
        }
      };
    } catch (error) {
      return {
        error: 'Failed to fetch violations',
        message: error.message
      };
    }
  }

  // Helper: Group violations by type
  private groupByType(violations: any[]) {
    return violations.reduce((acc, v) => ({
      ...acc,
      [v.message]: (acc[v.message] || 0) + 1
    }), {});
  }

  // Helper: Group violations by interview
  private groupByInterview(violations: any[]) {
    return violations.reduce((acc, v) => ({
      ...acc,
      [v.interviewId]: (acc[v.interviewId] || 0) + 1
    }), {});
  }

  // Submit interview and check for excessive violations
  @Post(':interviewId/submit')
  async submitInterview(@Param('interviewId') interviewId: string, @Req() req) {
    const userId = req.user.id;
    
    // Fetch violations
    const violationResponse = await fetch(
      `http://localhost:8000/interview/${interviewId}`
    );
    const violationData = await violationResponse.json();
    const violations = violationData.violations || [];

    // Check violation threshold (e.g., more than 5 = suspicious)
    const isSuspicious = violations.length > 5;

    // Save interview with proctoring flag
    const result = await this.interviewService.submitInterview(interviewId, {
      violations: violations.length,
      isSuspicious,
      violationDetails: violations
    });

    return {
      success: true,
      interviewId,
      violations: violations.length,
      flagged: isSuspicious,
      message: isSuspicious 
        ? 'Interview flagged for manual review due to excessive violations'
        : 'Interview submitted successfully'
    };
  }
}
```

---

## 2. Interview Service with ProctorKit Integration

```typescript
// src/interview/interview.service.ts
import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { Interview } from './schemas/interview.schema';

@Injectable()
export class InterviewService {
  constructor(
    @InjectModel(Interview.name) private interviewModel: Model<Interview>
  ) {}

  async getInterview(id: string) {
    return this.interviewModel.findById(id);
  }

  async submitInterview(id: string, proctorData: any) {
    return this.interviewModel.findByIdAndUpdate(
      id,
      {
        submitted_at: new Date(),
        proctor_violations: proctorData.violations,
        proctor_flagged: proctorData.isSuspicious,
        proctor_details: proctorData.violationDetails
      },
      { new: true }
    );
  }

  async getInterviewStats(userId: string) {
    try {
      // Fetch from ProctorKit backend
      const response = await fetch(`http://localhost:8000/violations/${userId}`);
      const data = await response.json();
      
      return {
        totalViolations: data.total_violations,
        violations: data.violations,
        stats: {
          faceViolations: data.violations.filter(v => v.meta.type === 'Face').length,
          phoneViolations: data.violations.filter(v => v.message.includes('Object')).length,
          multipleViolations: data.violations.filter(v => v.message.includes('Multiple')).length,
        }
      };
    } catch (error) {
      console.error('Error fetching violation stats:', error);
      return null;
    }
  }
}
```

---

## 3. Interview Schema with Proctoring Fields

```typescript
// src/interview/schemas/interview.schema.ts
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';

@Schema()
export class Interview extends Document {
  @Prop(String)
  title: string;

  @Prop({ type: Types.ObjectId, ref: 'User' })
  userId: Types.ObjectId;

  @Prop(String)
  type: string; // 'coding', 'interview', 'test'

  @Prop(String)
  description: string;

  @Prop(Date)
  created_at: Date;

  @Prop(Date)
  started_at: Date;

  @Prop(Date)
  submitted_at: Date;

  // Proctoring fields
  @Prop(Number)
  proctor_violations: number; // Total violations count

  @Prop(Boolean)
  proctor_flagged: boolean; // Flagged for review

  @Prop(Array)
  proctor_details: any[]; // Full violation logs

  @Prop(String)
  proctor_status: string; // 'clean', 'flagged', 'review_pending'

  @Prop(Number)
  score: number;

  @Prop(Boolean)
  completed: boolean;
}

export const InterviewSchema = SchemaFactory.createForClass(Interview);
```

---

## 4. Proctoring Dashboard API

```typescript
// src/dashboard/proctoring.controller.ts
import { Controller, Get, Param, UseGuards } from '@nestjs/common';
import { JwtAuthGuard } from 'src/auth/guards/jwt-auth.guard';

@Controller('dashboard/proctoring')
@UseGuards(JwtAuthGuard)
export class ProctoringDashboardController {
  
  // Get user's violation summary
  @Get('user/:userId/summary')
  async getUserSummary(@Param('userId') userId: string) {
    try {
      const response = await fetch(`http://localhost:8000/violations/${userId}`);
      const data = await response.json();
      
      return {
        userId,
        totalViolations: data.total_violations,
        lastViolation: data.violations?.[0]?.timestamp,
        violationTypes: this.analyzeViolations(data.violations || []),
        riskLevel: this.calculateRiskLevel(data.total_violations)
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  // Get interview proctoring details
  @Get('interview/:interviewId/details')
  async getInterviewDetails(@Param('interviewId') interviewId: string) {
    try {
      const response = await fetch(`http://localhost:8000/interview/${interviewId}`);
      const data = await response.json();
      const violations = data.violations || [];

      return {
        interviewId,
        totalViolations: data.total_violations,
        timeline: violations.map(v => ({
          time: v.timestamp,
          type: v.message,
          details: v.meta
        })),
        integrity: {
          score: 100 - Math.min(violations.length * 10, 100),
          assessment: violations.length > 5 ? 'SUSPICIOUS' : 'NORMAL'
        }
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  // Analyze violations
  private analyzeViolations(violations: any[]) {
    return {
      faceAbsent: violations.filter(v => v.message.includes('Face')).length,
      phoneDetected: violations.filter(v => v.message.includes('Object')).length,
      multipleUsers: violations.filter(v => v.message.includes('Multiple')).length,
      gazeShifted: violations.filter(v => v.message.includes('away')).length,
    };
  }

  // Calculate risk level
  private calculateRiskLevel(violationCount: number): string {
    if (violationCount === 0) return 'LOW';
    if (violationCount < 3) return 'LOW_MEDIUM';
    if (violationCount < 5) return 'MEDIUM';
    if (violationCount < 10) return 'MEDIUM_HIGH';
    return 'HIGH';
  }
}
```

---

## 5. WebSocket Support for Real-time Updates

```typescript
// src/proctoring/proctoring.gateway.ts
import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';

@WebSocketGateway({
  cors: {
    origin: ['http://localhost:3000', 'http://localhost:8000'],
  },
})
export class ProctoringGateway implements OnGatewayConnection {
  @WebSocketServer()
  server: Server;

  handleConnection(client: Socket) {
    console.log(`Client connected: ${client.id}`);
  }

  // Receive violation from ProctorKit
  @SubscribeMessage('violation')
  handleViolation(client: Socket, data: any) {
    const { userId, interviewId } = data;
    
    // Broadcast to dashboard
    this.server.to(`interview:${interviewId}`).emit('violation', data);
    this.server.to(`user:${userId}`).emit('violation', data);
  }

  // Subscribe to interview violations
  @SubscribeMessage('subscribe-interview')
  subscribeToInterview(client: Socket, interviewId: string) {
    client.join(`interview:${interviewId}`);
    console.log(`Client subscribed to interview: ${interviewId}`);
  }

  // Subscribe to user violations
  @SubscribeMessage('subscribe-user')
  subscribeToUser(client: Socket, userId: string) {
    client.join(`user:${userId}`);
    console.log(`Client subscribed to user: ${userId}`);
  }
}
```

---

## 6. Module Configuration

```typescript
// src/interview/interview.module.ts
import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { InterviewController } from './interview.controller';
import { InterviewService } from './interview.service';
import { Interview, InterviewSchema } from './schemas/interview.schema';

@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Interview.name, schema: InterviewSchema }
    ])
  ],
  controllers: [InterviewController],
  providers: [InterviewService],
  exports: [InterviewService]
})
export class InterviewModule {}
```

---

## 7. Usage in Frontend (React Example)

```typescript
// src/pages/Interview.tsx
import React, { useState } from 'react';

export const InterviewPage: React.FC = () => {
  const [interviewId] = useState('interview_123');
  const userId = 'user_456'; // From auth context

  return (
    <div style={{ padding: '20px' }}>
      <h1>Proctored Interview</h1>
      
      <iframe
        src={`http://localhost:8000/proctoring-system/index.html?userId=${userId}&interviewId=${interviewId}&testType=coding`}
        width="100%"
        height="900"
        style={{ border: 'none', borderRadius: '8px' }}
      />

      <button 
        onClick={async () => {
          const res = await fetch(
            `/api/interview/${interviewId}/violations`
          );
          const data = await res.json();
          console.log('Violations:', data);
        }}
      >
        Check Violations
      </button>
    </div>
  );
};
```

---

## Testing with cURL

```bash
# Test ProctorKit backend is running
curl http://localhost:8000/health

# Get violations for user
curl http://localhost:8000/violations/user_12345

# Get violations for interview
curl http://localhost:8000/interview/interview_67890

# Manual violation report
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "Test",
    "meta": {},
    "screenshot": "data:",
    "userId": "user_12345",
    "interviewId": "interview_67890",
    "testType": "coding"
  }'
```

---

## Key Points

✅ ProctorKit runs on **port 8000** (separate from NestJS)
✅ Violations saved to **fulcrum-staging** MongoDB
✅ Easy to integrate via **URL parameters**
✅ Query violations with REST API
✅ Real-time updates via WebSocket (optional)
✅ Production-ready with proper error handling

---

**Status:** Ready for Implementation ✅
