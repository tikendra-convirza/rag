from fastapi import FastAPI, UploadFile, File as FastAPIFile, Form
from pydantic import BaseModel
from ...domain.ports import AskQuestionPort, IngestionPort, Answer
from ...domain.model import Citation
from src.application.rag_service import RagService
from pathlib import Path
app = FastAPI()

from src.adapters.outbound.retriever_llamaindex import LlamaindexRetriever
from src.adapters.outbound.generator_openai import LitellmGenerator
from src.application.rag_service import RagService
from src.adapters.inbound.ingestion import LlamaindexIngestionAdapter
from llama_index.core.indices import VectorStoreIndex
import logging
import sys
import os
from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter

import qdrant_client
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from src.adapters.llamaindex_utils import get_qdrant_vector_store, get_vectorstore_index
from src.logger import setup_logger

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5", cache_dir='./fastembed_weights')

url=os.getenv('QDRANT_DB_URL')
api_key=os.getenv('QDRANT_API_KEY')
url = 'http://localhost:6333'
vector_store = get_qdrant_vector_store(url=url, api_key=api_key, collection_name='rag_collection')
index = get_vectorstore_index(vector_store=vector_store)

transformations=[
    # SemanticSplitterNodeParser(embed_model=Settings.embed_model), 
                #  QuestionsAnsweredExtractor(),
    SentenceSplitter(chunk_size=312, chunk_overlap=50),
                 ] 
rag_service = RagService(
    retriever=LlamaindexRetriever(index=index),
    generator=LitellmGenerator(),
    ingester=LlamaindexIngestionAdapter(storage_dir='ingestion_files', transformations=transformations),
)

logger = setup_logger(__name__)

class AskRequest(BaseModel):
    query: str
    filters: dict

class AnswerDTO(BaseModel):
    text: str
    citations: list[Citation]

@app.post("/ask", response_model=AnswerDTO)
async def ask_endpoint(req: AskRequest) -> AnswerDTO:
    logger.info("Received ask request: %s", req)
    answer: Answer = await rag_service.ask(req.query, filters=req.filters)
    logger.debug("Answer generated: %s", answer)
    return AnswerDTO(**answer.__dict__)

@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = FastAPIFile(...),
    metadata: str = Form("{}")
):
    logger.info("Received ingest request for file: %s", file.filename)
    if file.content_type != "text/plain":
        return {"error": "Only text files are supported."}
    
    # Save file to disk using Path
    save_dir = Path("ingestion_files")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    # Parse metadata string to dict
    try:
        metadata_dict = eval(metadata)
        logger.debug("Parsed metadata: %s", metadata_dict)
        if not isinstance(metadata_dict, dict):
            metadata_dict = {}
    except Exception:
        logger.error("Failed to parse metadata: %s", metadata)
        metadata_dict = {}
    # Pass file path and metadata to rag_service.ingest
    await rag_service.ingest(filepath=save_path, metadata=metadata_dict)
    logger.info("Ingestion completed for file: %s", save_path)
    return {"status": "success"}
