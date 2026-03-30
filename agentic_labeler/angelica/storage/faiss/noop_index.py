from __future__ import annotations

"""No-op vector index for when RAG/vector indexing is disabled.

This provides the same interface as FaissVectorIndex but performs no operations,
allowing the system to work without vector indexing and retrieval.
"""

from typing import Any, Dict, List, Optional


class NoOpVectorIndex:
    """A no-op vector index that does nothing but maintains the same interface."""

    def __init__(self, *args, **kwargs):
        """Accept any arguments to maintain compatibility but ignore them."""
        pass

    def load(self) -> None:
        """No-op load."""
        pass

    def save(self) -> None:
        """No-op save."""
        pass

    def add_document(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """No-op add document."""
        pass

    def similarity_doc_ids(
        self,
        query: str,
        k: int = 5,
        exclude_doc_id: Optional[int] = None
    ) -> List[int]:
        """Return empty list since no indexing is performed."""
        return []

    @property
    def vs(self):
        """Return None for compatibility."""
        return None


