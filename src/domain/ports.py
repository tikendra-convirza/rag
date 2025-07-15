from abc import ABC, abstractmethod
from typing import List
from .model import BaseDocument, Answer

# ---------- inbound (driving) ----------

class InboundPort(ABC):
    ...
    
class AskQuestionPort(InboundPort):
    @abstractmethod
    async def ask(self, question: str) -> Answer: ...

class IngestionPort(InboundPort):
    @abstractmethod
    async def ingest(self, **kwargs) -> List[BaseDocument]: ...
# ---------- outbound (driven) ----------

class OutboundPort(ABC):
    ...

class RetrieveDocumentsPort(OutboundPort):
    @abstractmethod
    async def retrieve(self, query: str, **kwargs) -> List[BaseDocument]: ...

    @abstractmethod
    async def ingest(self, documents:list[BaseDocument], **kwargs) -> None: ...

class GenerateAnswerPort(OutboundPort):
    @abstractmethod
    async def generate(self, question: str, docs: List[BaseDocument]) -> Answer: ...

class StoreInteractionPort(OutboundPort):
    @abstractmethod
    async def save(self, question: str, answer: Answer, docs: List[BaseDocument]) -> None: ...
