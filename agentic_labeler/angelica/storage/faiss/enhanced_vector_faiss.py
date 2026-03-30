from __future__ import annotations

"""Enhanced FAISS-based vector index with improved matching and pattern learning.

Improvements:
1. Better similarity matching with configurable thresholds and scoring
2. Pattern storage and learning for unmatched cases
3. Pattern evolution based on accumulated examples
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from angelica.llm_client.llm import make_embeddings


@dataclass
class MatchResult:
    """Result of a similarity search with scoring information."""
    doc_id: int
    score: float
    metadata: Dict[str, Any]
    is_confident: bool  # Whether the match meets confidence threshold


@dataclass
class Pattern:
    """A learned pattern with examples and metadata."""
    pattern_id: str
    pattern_name: str
    description: str
    example_doc_ids: List[int]
    created_at: str
    updated_at: str
    confidence_score: float
    usage_count: int


class EnhancedFaissVectorIndex:
    """Enhanced FAISS index with better matching and pattern learning."""

    def __init__(
        self,
        index_dir: str = "vector_index",
        patterns_file: str = "learned_patterns.json",
        similarity_threshold: float = 0.7,
        min_confidence_threshold: float = 0.6,
    ):
        self.index_dir = index_dir
        self.patterns_file = os.path.join(index_dir, patterns_file)
        self.similarity_threshold = similarity_threshold
        self.min_confidence_threshold = min_confidence_threshold
        
        self._embeddings = make_embeddings()
        self._vs: Optional[FAISS] = None
        self._patterns: Dict[str, Pattern] = {}
        
        # Load patterns if they exist
        self._load_patterns()

    def load(self) -> None:
        """Load an existing FAISS index if present."""
        if os.path.isdir(self.index_dir) and os.listdir(self.index_dir):
            try:
                self._vs = FAISS.load_local(
                    self.index_dir,
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                # Index might not exist yet
                self._vs = None
        else:
            self._vs = None

    @property
    def vs(self) -> FAISS:
        """Underlying LangChain FAISS vector store (lazy-loaded)."""
        if self._vs is None:
            self.load()
        return self._vs  # type: ignore[return-value]

    def save(self) -> None:
        """Persist index and patterns to disk."""
        if self._vs is not None:
            self._vs.save_local(self.index_dir)
        self._save_patterns()

    def add_document(
        self,
        doc_id: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add one document to the index."""
        if self._vs is None:
            # First document → create index
            self._vs = FAISS.from_documents(
                [Document(page_content=text, metadata={"doc_id": doc_id, **(metadata or {})})],
                self._embeddings,
            )
        else:
            self._vs.add_documents(
                [Document(page_content=text, metadata={"doc_id": doc_id, **(metadata or {})})]
            )

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        exclude_doc_id: Optional[int] = None
    ) -> List[MatchResult]:
        """
        Return up to k most similar documents with confidence scores.
        
        Returns MatchResult objects that include:
        - doc_id: document identifier
        - score: similarity score (0-1, higher is better)
        - metadata: document metadata
        - is_confident: whether score meets confidence threshold
        """
        if self._vs is None:
            return []
        
        # Get similarity search with scores (FAISS returns distance, lower is better)
        docs_and_scores = self.vs.similarity_search_with_score(query, k=k + 5)
        
        results: List[MatchResult] = []
        for doc, distance in docs_and_scores:
            doc_id = int(doc.metadata.get("doc_id", 0))
            
            if exclude_doc_id is not None and doc_id == exclude_doc_id:
                continue
            
            # Convert distance to similarity score (0-1, higher is better)
            # For L2 distance, we use: similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance)
            
            is_confident = similarity_score >= self.similarity_threshold
            
            results.append(MatchResult(
                doc_id=doc_id,
                score=similarity_score,
                metadata=doc.metadata,
                is_confident=is_confident
            ))
            
            if len(results) >= k:
                break
        
        return results

    def similarity_doc_ids(
        self,
        query: str,
        k: int = 5,
        exclude_doc_id: Optional[int] = None
    ) -> List[int]:
        """Return up to k doc_ids most similar to the query text (backward compatible)."""
        results = self.similarity_search_with_scores(query, k, exclude_doc_id)
        return [r.doc_id for r in results]

    def get_best_matches(
        self,
        query: str,
        k: int = 5,
        exclude_doc_id: Optional[int] = None,
        require_confident: bool = False
    ) -> Tuple[List[MatchResult], bool]:
        """
        Get best matches with quality assessment.
        
        Returns:
        - List of MatchResult objects
        - Boolean indicating if any confident matches were found
        """
        results = self.similarity_search_with_scores(query, k, exclude_doc_id)
        
        if require_confident:
            results = [r for r in results if r.is_confident]
        
        has_confident_matches = any(r.is_confident for r in results)
        
        return results, has_confident_matches

    # Pattern Learning Methods

    def _load_patterns(self) -> None:
        """Load learned patterns from disk."""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    self._patterns = {
                        pid: Pattern(**pdata) for pid, pdata in data.items()
                    }
            except json.JSONDecodeError as e:
                print(f"Warning: Could not load patterns (corrupted JSON): {e}")
                print(f"  Backing up corrupted file and starting fresh")
                # Backup corrupted file
                backup_path = self.patterns_file + ".backup"
                try:
                    import shutil
                    shutil.copy(self.patterns_file, backup_path)
                    print(f"  Corrupted file backed up to: {backup_path}")
                except Exception:
                    pass
                self._patterns = {}
            except Exception as e:
                print(f"Warning: Could not load patterns: {e}")
                self._patterns = {}
        else:
            self._patterns = {}

    def _save_patterns(self) -> None:
        """Save learned patterns to disk."""
        import numpy as np
        
        def convert_to_python_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python_types(item) for item in obj]
            return obj
        
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.patterns_file, 'w') as f:
            data = {pid: asdict(p) for pid, p in self._patterns.items()}
            # Convert any numpy types to Python types
            data = convert_to_python_types(data)
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _normalize_pattern_name(self, name: str) -> str:
        """Normalize pattern name for comparison."""
        import re
        # Convert to lowercase, remove extra spaces, remove special chars
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_similar_pattern(
        self,
        pattern_name: str,
        description: str = "",
        name_threshold: float = 0.8,
        description_threshold: float = 0.6
    ) -> Optional[Pattern]:
        """
        Find an existing pattern that's similar to the given name and description.
        
        Uses both name and description for better matching to avoid creating
        duplicate patterns with slightly different names.
        """
        normalized_name = self._normalize_pattern_name(pattern_name)
        
        best_match = None
        best_score = 0.0
        
        for pattern in self._patterns.values():
            normalized_existing = self._normalize_pattern_name(pattern.pattern_name)
            
            # Exact name match
            if normalized_name == normalized_existing:
                return pattern
            
            # Calculate name similarity
            name_similarity = 0.0
            if normalized_name in normalized_existing or normalized_existing in normalized_name:
                # Substring match
                len_ratio = min(len(normalized_name), len(normalized_existing)) / max(len(normalized_name), len(normalized_existing))
                name_similarity = len_ratio
            else:
                # Word-based similarity
                name_similarity = self._calculate_similarity(normalized_name, normalized_existing)
            
            # Calculate description similarity if provided
            description_similarity = 0.0
            if description and pattern.description:
                description_similarity = self._calculate_similarity(description, pattern.description)
            
            # Combined score: prioritize name but consider description
            combined_score = name_similarity * 0.7 + description_similarity * 0.3
            
            # Check if this is a good match
            if name_similarity >= name_threshold or (name_similarity >= 0.5 and description_similarity >= description_threshold):
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = pattern
        
        return best_match
    
    def create_pattern(
        self,
        pattern_id: str,
        pattern_name: str,
        description: str,
        initial_doc_id: int,
        confidence_score: float = 0.5
    ) -> Pattern:
        """
        Create a new pattern from an unmatched or low-confidence case.
        
        This is called when the system encounters a document that doesn't
        match existing patterns well. It checks for similar existing patterns
        first to avoid duplicates.
        """
        from datetime import datetime, timezone
        
        # Check if a similar pattern already exists
        similar_pattern = self._find_similar_pattern(pattern_name, description)
        if similar_pattern:
            # Update the existing pattern instead of creating a new one
            print(f"  → Grouping with existing pattern: {similar_pattern.pattern_name}")
            updated = self.update_pattern(
                similar_pattern.pattern_id,
                initial_doc_id,
                confidence_score
            )
            return updated if updated else similar_pattern
        
        now = datetime.now(timezone.utc).isoformat()
        
        pattern = Pattern(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            description=description,
            example_doc_ids=[initial_doc_id],
            created_at=now,
            updated_at=now,
            confidence_score=confidence_score,
            usage_count=1
        )
        
        self._patterns[pattern_id] = pattern
        self._save_patterns()
        
        return pattern

    def update_pattern(
        self,
        pattern_id: str,
        new_doc_id: int,
        new_confidence: Optional[float] = None
    ) -> Optional[Pattern]:
        """
        Update an existing pattern with a new example.
        
        This evolves the pattern over time as more examples are added.
        """
        if pattern_id not in self._patterns:
            return None
        
        from datetime import datetime, timezone
        
        pattern = self._patterns[pattern_id]
        
        # Add new example if not already present
        if new_doc_id not in pattern.example_doc_ids:
            pattern.example_doc_ids.append(new_doc_id)
        
        # Update confidence score (moving average)
        if new_confidence is not None:
            pattern.confidence_score = (
                pattern.confidence_score * 0.8 + new_confidence * 0.2
            )
        
        pattern.usage_count += 1
        pattern.updated_at = datetime.now(timezone.utc).isoformat()
        
        self._save_patterns()
        
        return pattern

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve a pattern by ID."""
        return self._patterns.get(pattern_id)

    def get_all_patterns(self) -> List[Pattern]:
        """Get all learned patterns."""
        return list(self._patterns.values())

    def find_pattern_for_label(self, label_name: str) -> Optional[Pattern]:
        """Find a pattern matching a given label name."""
        for pattern in self._patterns.values():
            if pattern.pattern_name == label_name:
                return pattern
        return None

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        if not self._patterns:
            return {
                "total_patterns": 0,
                "total_examples": 0,
                "avg_confidence": 0.0,
                "avg_usage": 0.0
            }
        
        total_examples = sum(len(p.example_doc_ids) for p in self._patterns.values())
        avg_confidence = sum(p.confidence_score for p in self._patterns.values()) / len(self._patterns)
        avg_usage = sum(p.usage_count for p in self._patterns.values()) / len(self._patterns)
        
        return {
            "total_patterns": len(self._patterns),
            "total_examples": total_examples,
            "avg_confidence": avg_confidence,
            "avg_usage": avg_usage,
            "patterns": [
                {
                    "id": p.pattern_id,
                    "name": p.pattern_name,
                    "examples": len(p.example_doc_ids),
                    "confidence": p.confidence_score,
                    "usage": p.usage_count
                }
                for p in sorted(self._patterns.values(), key=lambda x: x.usage_count, reverse=True)
            ]
        }


