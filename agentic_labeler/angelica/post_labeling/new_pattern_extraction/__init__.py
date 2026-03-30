"""New pattern extraction module.

This module provides functionality to discover new patterns from labeled data
that doesn't fit existing pattern categories.
"""

from .pattern_extractor import (
    PatternExtractor,
    NewPattern,
    PatternExtractionResult,
)

__all__ = [
    "PatternExtractor",
    "NewPattern",
    "PatternExtractionResult",
]


