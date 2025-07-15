from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from ...domain.model import Answer, BaseDocument
from ...domain.ports import StoreInteractionPort

class MongoInteractionRepo(StoreInteractionPort):
    def __init__(self, client: AsyncIOMotorClient, db_name: str = "rag") -> None:
        self._coll = client[db_name]["interactions"]

    async def save(self, question: str, answer: Answer, docs: List[BaseDocument]) -> None:
        await self._coll.insert_one({
            "question": question,
            "answer": answer.text,
            "citations": [c.__dict__ for c in answer.citations],
            "doc_ids": [d.id for d in docs],
        })
