from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from src.logger import setup_logger

logger = setup_logger(__name__)

class VectorDatabase(ABC):
    """
    Abstract base class for vector databases.
    """

    @abstractmethod
    def add_documents(self, 
                      documents: List[str], 
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        Embed and add documents to the vector store.
        Returns list of document IDs.
        """
        pass

    @abstractmethod
    def add_embeddings(self, 
                       embeddings: List[np.ndarray], 
                       metadata: Optional[List[Dict[str, Any]]] = None,
                       ids: Optional[List[str]] = None) -> List[str]:
        """
        Add raw vector embeddings directly to the store.
        """
        pass

    @abstractmethod
    def similarity_search(self, 
                          query: Union[str, np.ndarray], 
                          top_k: int = 5, 
                          metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Return top_k similar documents based on embedding similarity.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents or embeddings by their IDs.
        """
        pass

    @abstractmethod
    def update_metadata(self, 
                        id: str, 
                        metadata: Dict[str, Any]) -> None:
        """
        Update metadata for a specific document/vector.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all entries from the database.
        """
        pass

    @abstractmethod
    def persist(self, path: str) -> None:
        """
        Save current state to disk or external store.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a previously saved vector store from disk.
        """
        pass