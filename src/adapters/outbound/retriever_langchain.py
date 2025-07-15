from typing import List
import asyncpg
from ...domain.model import Document
from ...domain.ports import RetrieveDocumentsPort

class PGVectorRetriever(RetrieveDocumentsPort):
    def __init__(self, pool: asyncpg.Pool, table: str = "embeddings") -> None:
        self._pool = pool
        self._table = table

    async def retrieve(self, query: str, *, k: int = 8) -> List[Document]:
        # ‑‑ embed query, run ANN search, map rows → Document …
        raise NotImplementedError
