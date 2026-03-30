"""Ray actor for thread-safe storage operations.

This module provides a Ray actor that handles all SQLite and FAISS operations
in a thread-safe manner. It batches FAISS index updates for efficiency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for parallel processing. "
        "Install it with: pip install 'angelica[parallel]' or pip install ray"
    )

from pydantic import BaseModel

from angelica.storage.sqlite.store_sqlite import SQLiteStore
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.noop_index import NoOpVectorIndex


@ray.remote
class RayStorageActor:
    """Ray actor for centralized, thread-safe storage operations.
    
    This actor serializes all writes to SQLite and FAISS, ensuring data consistency
    in a parallel processing environment. It batches FAISS index updates for efficiency.
    
    Args:
        db_path: Path to SQLite database
        index_dir: Directory for FAISS index
        schema: Pydantic model class for labels
        store_spec: Storage specification (optional)
        batch_size: Number of documents to batch before flushing FAISS index
        enable_rag: Whether to enable RAG/vector indexing (default: True)
    
    Example:
        >>> storage = RayStorageActor.remote(
        ...     db_path="labels.db",
        ...     index_dir="vector_index",
        ...     schema=MyLabelSchema
        ... )
        >>> doc_id = ray.get(storage.add_document.remote("code", "source"))
    """
    
    def __init__(
        self,
        db_path: str,
        index_dir: str,
        schema: Type[BaseModel],
        store_spec: Optional[Any] = None,
        batch_size: int = 10,
        enable_rag: bool = True,
    ):
        """Initialize the storage actor."""
        from angelica.models.config import StoreSpec
        
        self.store = SQLiteStore(
            db_path=db_path,
            schema=schema,
            store_spec=store_spec or StoreSpec(),
        )
        
        self.enable_rag = enable_rag
        
        # Use NoOpVectorIndex if RAG is disabled
        if not enable_rag:
            self.index: Union[FaissVectorIndex, NoOpVectorIndex] = NoOpVectorIndex()
        else:
            self.index = FaissVectorIndex(index_dir=index_dir)
            self.index.load()
        
        # Batching for FAISS updates
        self.batch_size = batch_size
        self._pending_docs: List[tuple[int, str, Dict[str, Any]]] = []
    
    def add_document(self, text: str, source: Optional[str] = None) -> int:
        """Add a document to SQLite and return its doc_id.
        
        Args:
            text: Document text content
            source: Optional source identifier
            
        Returns:
            Document ID assigned by SQLite
        """
        return self.store.add_document(text, source)
    
    def save_label(self, doc_id: int, agent_id: str, label: BaseModel) -> None:
        """Save a per-agent label to SQLite.
        
        Args:
            doc_id: Document ID
            agent_id: Agent identifier
            label: Label object (Pydantic model)
        """
        self.store.save_label(doc_id, agent_id, label)
    
    def save_final_label(self, doc_id: int, decided_by: str, label: BaseModel) -> None:
        """Save the final label to SQLite.
        
        Args:
            doc_id: Document ID
            decided_by: Decision method (e.g., "agreement" or "adjudicator:agent_id")
            label: Final label object (Pydantic model)
        """
        self.store.save_final_label(doc_id, decided_by, label)
    
    def add_to_index(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the FAISS index (batched).
        
        Documents are batched and the index is flushed when batch_size is reached.
        Call flush_and_save() at the end to ensure all documents are indexed.
        
        Args:
            doc_id: Document ID
            text: Text to embed and index
            metadata: Optional metadata to store with the document
        """
        # Skip indexing if RAG is disabled
        if not self.enable_rag:
            return
            
        self._pending_docs.append((doc_id, text, metadata or {}))
        
        if len(self._pending_docs) >= self.batch_size:
            self._flush_index()
    
    def _flush_index(self) -> None:
        """Flush pending documents to the FAISS index."""
        if not self._pending_docs:
            return
        
        for doc_id, text, metadata in self._pending_docs:
            self.index.add_document(doc_id, text, metadata)
        
        self._pending_docs.clear()
        self.index.save()
    
    def flush_and_save(self) -> None:
        """Flush any pending documents and save the FAISS index.
        
        This should be called at the end of parallel processing to ensure
        all documents are indexed and persisted.
        """
        self._flush_index()
    
    def get_index_handle(self) -> Union[FaissVectorIndex, NoOpVectorIndex]:
        """Get the FAISS index for read operations.
        
        Note: This is intended for read-only operations. Write operations
        should go through add_to_index() to maintain batching.
        
        Returns:
            The FAISS vector index instance (or NoOpVectorIndex if RAG is disabled)
        """
        return self.index


