"""FAISS-based vector storage with enhanced matching and pattern learning."""

from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.enhanced_vector_faiss import (
    EnhancedFaissVectorIndex,
    MatchResult,
    Pattern,
)
from angelica.storage.faiss.noop_index import NoOpVectorIndex

__all__ = [
    "FaissVectorIndex",
    "EnhancedFaissVectorIndex",
    "MatchResult",
    "Pattern",
    "NoOpVectorIndex",
]
