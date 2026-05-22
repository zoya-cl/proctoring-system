from motor.motor_asyncio import AsyncIOMotorClient
from config.settings import CONFIG
from bson import ObjectId

class Database:
    def __init__(self):
        self.client = AsyncIOMotorClient(CONFIG["MONGODB_URI"])
        self.db = self.client[CONFIG["DB_NAME"]]
        self.violations = self.db[CONFIG["COLLECTION_NAME"]]

    async def save_violation(self, violation_data):
        """Save a violation report with user/interview context"""
        return await self.violations.insert_one(violation_data)

    async def get_violations_by_user(self, user_id: str, limit: int = 100):
        """Get all violations for a specific user"""
        cursor = self.violations.find({"userId": user_id}).sort("timestamp", -1).limit(limit)
        violations = []
        async for violation in cursor:
            violation["_id"] = str(violation["_id"])
            violations.append(violation)
        return violations

    async def get_violations_by_interview(self, interview_id: str):
        """Get all violations for a specific interview/test"""
        cursor = self.violations.find({"interviewId": interview_id}).sort("timestamp", -1)
        violations = []
        async for violation in cursor:
            violation["_id"] = str(violation["_id"])
            violations.append(violation)
        return violations

    async def get_user_interview_violations(self, user_id: str, interview_id: str):
        """Get violations for a specific user in a specific interview"""
        cursor = self.violations.find({
            "userId": user_id,
            "interviewId": interview_id
        }).sort("timestamp", -1)
        violations = []
        async for violation in cursor:
            violation["_id"] = str(violation["_id"])
            violations.append(violation)
        return violations

    async def get_violations_by_type(self, test_type: str):
        """Get all violations for a test type (coding, interview, test)"""
        cursor = self.violations.find({"testType": test_type}).sort("timestamp", -1)
        violations = []
        async for violation in cursor:
            violation["_id"] = str(violation["_id"])
            violations.append(violation)
        return violations

    async def create_indexes(self):
        """Create database indexes for faster queries"""
        await self.violations.create_index("userId")
        await self.violations.create_index("interviewId")
        await self.violations.create_index("testType")
        await self.violations.create_index("timestamp")
        await self.violations.create_index([("userId", 1), ("interviewId", 1)])

db = Database()
