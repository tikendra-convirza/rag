from src.adapters.outbound.retriever_llamaindex import LlamaindexRetriever
from src.adapters.outbound.generator_openai import OpenAIChatGenerator
from src.adapters.outbound.store_mongo import MongoInteractionRepo
from src.application.rag_service import RagService
from src.adapters.inbound.ingestion import LlamaindexIngestionAdapter
from src.adapters.inbound.rest import create_app
from llama_index.core.indices import VectorStoreIndex
import logging
import sys
import os
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser

import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from src.logger import setup_logger

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5", cache_dir='./fastembed_weights')
logger = setup_logger(__name__)
async def init_app():
    logger.info("Initializing application")
    index = VectorStoreIndex()
    transformations=[SemanticSplitterNodeParser(embed_model=Settings.embed_model), QuestionsAnsweredExtractor()]
    rag = RagService(
        retriever=LlamaindexRetriever(index=index),
        generator=OpenAIChatGenerator(),
        ingester=LlamaindexIngestionAdapter(storage_dir='ingestion_files', transformations=transformations),
    )
    logger.info("Application initialized")
    return create_app(rag)

# uvicorn main:init_app --factory
