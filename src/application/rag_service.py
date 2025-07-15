from typing import Final, List, Any
from ..domain.model import BaseDocument, Answer, File
from ..domain.ports import (
    AskQuestionPort,
    RetrieveDocumentsPort,
    GenerateAnswerPort,
    IngestionPort,
    StoreInteractionPort,
)
from pathlib import Path
from functools import cache
from src.logger import setup_logger

logger = setup_logger(__name__)

@cache
def get_retriever(org_id:str)-> RetrieveDocumentsPort:
    ...

class RagService:
    """Pure domain logic; knows nothing about HTTP, LangChain, or databases."""

    _TOP_K: Final[int] = 8

    def __init__(
        self,
        ingester:IngestionPort,
        retriever: RetrieveDocumentsPort,
        generator: GenerateAnswerPort,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._ingester = ingester

    async def ask(self, question: str, filters:dict) -> Answer:
        logger.info("RagService.ask called with question='%s' filters=%s", question, filters)
        docs: List[BaseDocument] = await self._retriever.retrieve(question, filters=filters)
        logger.info('Retrieved documents: %d', len(docs))
        answer: Answer = await self._generator.generate(question, docs)
        logger.debug("Generated answer: %s", answer)
        return answer

    async def ingest(self, **kwargs) -> None:
        logger.info("RagService.ingest called with kwargs=%s", kwargs)
        docs = await self._ingester.ingest(**kwargs)
        if not docs:
            logger.error("No documents ingested")
            raise ValueError("No documents ingested")
        logger.info('Ingested documents: %d', len(docs))
        await self._retriever.ingest(docs, **kwargs)