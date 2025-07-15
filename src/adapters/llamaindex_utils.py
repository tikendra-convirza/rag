from llama_index.core.indices import VectorStoreIndex
import os
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from qdrant_client import AsyncQdrantClient
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_qdrant_vector_store(
    url: str,
    collection_name: str,
    api_key: str | None = None,
) -> BasePydanticVectorStore:
    client = qdrant_client.QdrantClient(
        
        # location=":memory:",
        api_key=api_key or os.getenv('QDRANT_API_KEY'), url=url,
        # check_compatibility=False
    )
    aclient = qdrant_client.AsyncQdrantClient(
        # location=":memory:",
        api_key=api_key or os.getenv('QDRANT_API_KEY'), url=url,
        # check_compatibility=False
    )
    vector_store = QdrantVectorStore(client=client,aclient=aclient, collection_name=collection_name, enable_hybrid=True,)
    return vector_store

def get_vectorstore_index(vector_store:BasePydanticVectorStore)-> VectorStoreIndex:

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        storage_context=storage_context,
        vector_store=vector_store,
    )

    logger.debug("Index created")
    return index