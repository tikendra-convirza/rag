from ...domain.ports import IngestionPort
from ...domain.model import File, BaseDocument
from typing import Any
from pathlib import Path
from functools import partial
from llama_index.core.readers.file.base import default_file_metadata_func
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document  
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    TransformComponent,
)
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from datetime import datetime
from src.logger import setup_logger

logger = setup_logger(__name__)

def now() -> str:
    """Returns the current timestamp in ISO format."""
    return datetime.now().isoformat()

def llamadocs_to_docs(documets:list[Document]) -> list[BaseDocument]:
    docs = []
    for doc in documets:
        docs.append(
            BaseDocument(
                id=doc.id_,
                text=doc.text,
                score= doc.score if hasattr(doc, 'score') else 0,
                metadata=doc.metadata,
                embedding=doc.embedding
            )
        )
    return docs

def add_metadata(input_file: str, x: dict, metadata_fn=default_file_metadata_func, fs=None) -> dict:
    metadata = metadata_fn(file_path=input_file, fs=fs)
    metadata.update(x)
    return metadata


def get_documents(filepath: Path, additional_metadata: dict=None, **kwargs) -> list[Document]:
    metadata_fn = kwargs.pop('file_metadata', default_file_metadata_func)

    if additional_metadata:
        metadata_fn = partial(add_metadata, x=additional_metadata, metadata_fn=metadata_fn)


    documents = SimpleDirectoryReader(input_files=[filepath.absolute().as_posix()], file_metadata=metadata_fn, **kwargs).load_data()

    
    return documents

class LlamaindexIngestionAdapter(IngestionPort):
    def __init__(self, storage_dir: Path|str, transformations: list[TransformComponent]|None = None, **ingestion_pipeline_kwargs) -> None:
        self.storage_dir = Path(storage_dir) if isinstance(storage_dir, str) else storage_dir
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._ingestion_pipeline = IngestionPipeline(
            transformations=transformations or [],
            **ingestion_pipeline_kwargs

        )
        logger.info("Initialized LlamaindexIngestionAdapter with storage_dir=%s", self.storage_dir)

    def _save_file(self, file_obj: Any, original_filename: str) -> Path:
        save_path = self.storage_dir / Path(original_filename).name
        file_obj.seek(0)
        with open(save_path, 'wb') as f:
            f.write(file_obj.read())
        logger.debug("Saving file: %s", original_filename)
        return save_path

    async def ingest(self, filepath: Path | None = None, metadata:dict|None=None, **kwargs) -> list[BaseDocument]:
        logger.info("Ingest called with filepath=%s metadata=%s", filepath, metadata)
        """
        Ingest any type of data: file, path, text, etc.
        """

        if filepath is not None and filepath.exists():
            logger.debug("Filepath exists: %s", filepath)
            docs = get_documents(filepath=filepath,additional_metadata=metadata, **kwargs)
            docs = self._ingestion_pipeline.run(documents=docs)
            logger.info("Ingested %d documents", len(docs))
            return docs

        return []

