from motor.motor_asyncio import AsyncIOMotorClient
from config.settings import CONFIG

class Database:
    def __init__(self):
        self.client = AsyncIOMotorClient(CONFIG["MONGODB_URI"])
        self.db = self.client[CONFIG["DB_NAME"]]
        self.violations = self.db[CONFIG["COLLECTION_NAME"]]

    async def save_violation(self, violation_data):
        return await self.violations.insert_one(violation_data)

db = Database()
