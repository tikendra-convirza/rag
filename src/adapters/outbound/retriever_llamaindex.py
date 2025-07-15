from typing import List
import asyncpg
from ...domain.model import BaseDocument
from ...domain.ports import RetrieveDocumentsPort
from llama_index.core import vector_stores
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.schema import BaseNode, TransformComponent, NodeWithScore, TextNode
from typing import Any, Generator, List, Optional, Sequence, Union
from llama_index.core.vector_stores.types import VectorStoreQuery, MetadataFilters, MetadataFilter, FilterCondition

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
import logging
from functools import lru_cache
from src.logger import setup_logger

logger = setup_logger(__name__)


class LlamaindexRetriever(RetrieveDocumentsPort):
    def __init__(self, index:VectorStoreIndex, similarity_top_k:int=3, filters:MetadataFilters=None, **kwargs) -> None:
        self._index = index
        self.similarity_top_k = similarity_top_k
        self.filters = filters
        self.kwargs = kwargs
        self._retriever = self._get_retriever(filters)
        logger.info("Initialized LlamaindexRetriever with top_k=%s filters=%s", similarity_top_k, filters)

    def _get_retriever(self, filters:dict=None):
        # Accept filters as dict, convert to MetadataFilters if needed
        if filters and isinstance(filters, dict):
            filters = MetadataFilters(filters=[
                MetadataFilter(key=k, value=v) for k, v in filters.items()
            ]) # type: ignore
        logger.debug("Getting retriever with filters: %s", filters)
        return self._index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=filters,
            kwargs=self.kwargs
        )
    def _dict_to_tuple(self, d: Optional[dict]) -> Optional[tuple]:
        if d is None:
            return None
        # Recursively convert dict to tuple for hashability
        def convert(value):
            if isinstance(value, dict):
                return tuple(sorted((k, convert(v)) for k, v in value.items()))
            elif isinstance(value, list):
                return tuple(convert(v) for v in value)
            else:
                return value
        return tuple(sorted((k, convert(v)) for k, v in d.items()))

    @lru_cache(maxsize=512*8)
    def get_retriever(self, filters_tuple: Optional[tuple] = None):
        """
        Return a cached retriever for given filters.
        filters_tuple: tuple of (key, value) pairs or None
        """
        # Convert tuple back to dict for _get_retriever
        def tuple_to_dict(t):
            if t is None:
                return None
            return {k: tuple_to_dict(v) if isinstance(v, tuple) and len(v) and isinstance(v[0], tuple) else v for k, v in t}
        filters = tuple_to_dict(filters_tuple)
        return self._get_retriever(filters)


    def set_filters(self, filters: dict = None):
        """Set new filters and update retriever."""
        filters_tuple = tuple(sorted(filters.items())) if filters else None
        self.filters = filters
        self._retriever = self.get_retriever(filters_tuple)
        logger.info("Setting new filters: %s", filters)

    async def retrieve(self, query: str, filters: dict = None) -> List[Document]:
        logger.info("Retrieving for query: '%s' with filters: %s", query, filters)
        """Retrieve with optional filters (uses cached retriever if filters provided)."""
        if filters is None or filters == {}:
            retriever = self._retriever
        else:
            filters_tuple = self._dict_to_tuple(filters)
            retriever = self.get_retriever(filters_tuple)
        nodes = await retriever.aretrieve(query)
        return self.parse_to_basedocuments(nodes)
    
    def parse_to_basedocuments(self, nodes:list[NodeWithScore])-> list[Document]:
        docs = []
        for node in nodes:
            docs.append(
                Document(
                    text = node.text,
                    metadata = node.metadata,
                    id = node.id_,
                    score = node.score,
                    embedding = node.embedding
                )
            )
        return docs
    
    def parse_to_nodes(self, documents:list[BaseDocument])-> Sequence[NodeWithScore]:
        docs = []
        for node in documents:
            docs.append(
                
                    TextNode(
                    text = node.text,
                    metadata = node.metadata,
                    embedding = node.embedding,)
            )
        return docs

    async def ingest(self, documents: list[BaseDocument], **kwargs) -> List[str]:
        logger.info("Ingesting %d documents", len(documents))
        """
        Ingest nodes/documents into the vector database.
        """
        print(type(documents))
        nodes = self.parse_to_nodes(documents)
        if nodes:
            await self._index.ainsert_nodes(nodes, **kwargs)