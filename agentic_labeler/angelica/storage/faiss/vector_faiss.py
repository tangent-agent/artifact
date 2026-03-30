from __future__ import annotations

"""FAISS-based vector index for retrieval.

We store embeddings for *documents* (e.g. Java test bodies) so we can retrieve
similar past documents and include them as examples when labeling new ones.
"""

import os
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from angelica.llm_client.llm import make_embeddings


class FaissVectorIndex:
    """FAISS index persisted to a local directory."""

    def __init__(self, index_dir: str = "vector_index"):
        self.index_dir = index_dir
        self._embeddings = make_embeddings()
        self._vs: Optional[FAISS] = None

    def load(self) -> None:
        """
        Load an existing FAISS index if present.
        Otherwise, defer index creation until the first document is added.
        """
        if os.path.isdir(self.index_dir) and os.listdir(self.index_dir):
            self._vs = FAISS.load_local(
                self.index_dir,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # IMPORTANT:
            # Do NOT create FAISS index with zero documents.
            # We create it lazily on first add_document().
            self._vs = None

    @property
    def vs(self) -> FAISS:
        """Underlying LangChain FAISS vector store (lazy-loaded)."""
        if self._vs is None:
            self.load()
        return self._vs  # type: ignore[return-value]

    def save(self) -> None:
        """Persist index to disk."""
        self.vs.save_local(self.index_dir)

    def add_document(self, doc_id: int, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
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

    def similarity_doc_ids(self, query: str, k: int = 5, exclude_doc_id: Optional[int] = None) -> List[int]:
        """Return up to k doc_ids most similar to the query text."""
        if self._vs is None:
            return []
        docs = self.vs.similarity_search(query, k=k + 5)  # oversample so we can exclude the current doc
        out: List[int] = []
        for d in docs:
            doc_id = int(d.metadata.get("doc_id"))
            if exclude_doc_id is not None and doc_id == exclude_doc_id:
                continue
            if doc_id not in out:
                out.append(doc_id)
            if len(out) >= k:
                break
        return out
