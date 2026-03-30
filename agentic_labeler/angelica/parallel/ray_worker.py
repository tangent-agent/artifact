"""Ray worker for parallel document labeling.

This module provides a stateless Ray worker that processes documents in parallel,
using async/await to parallelize Labeler A and B calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for parallel processing. "
        "Install it with: pip install 'angelica[parallel]' or pip install ray"
    )

from pydantic import BaseModel

from angelica.agents.agents import LabelerAgent, AdjudicatorAgent
from angelica.models.config import AgenticConfig, LabelingContext
from angelica.prompts.prompts import default_examples_formatter
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.noop_index import NoOpVectorIndex
from angelica.storage.sqlite.store_sqlite import SQLiteStore

logger = logging.getLogger(__name__)


@ray.remote
class RayLabelingWorker:
    """Stateless Ray worker for parallel document labeling.
    
    This worker processes documents independently, using async/await to parallelize
    calls to Labeler A and Labeler B. It does not perform any storage operations
    directly; instead, it returns results to the orchestrator.
    
    Args:
        db_path: Path to SQLite database (for retrieval only)
        index_dir: Directory for FAISS index (for retrieval only)
        config: Agentic labeling configuration
        context: Labeling context
        agent_a_id: Identifier for Labeler A
        agent_b_id: Identifier for Labeler B
        adjudicator_id: Identifier for adjudicator
        temperature: LLM temperature
    
    Example:
        >>> worker = RayLabelingWorker.remote(
        ...     db_path="labels.db",
        ...     index_dir="vector_index",
        ...     config=config,
        ...     context=context
        ... )
        >>> result = ray.get(worker.label_document_async.remote("code", "source", 1))
    """
    
    def __init__(
        self,
        db_path: str,
        index_dir: str,
        config: AgenticConfig,
        context: LabelingContext,
        agent_a_id: str = "labeler_A",
        agent_b_id: str = "labeler_B",
        adjudicator_id: str = "adjudicator_1",
        temperature: float = 0.1,
        validation_fn: Optional[Callable[[BaseModel, str], List[str]]] = None,
    ):
        """Initialize the worker with read-only access to storage.
        
        Args:
            validation_fn: Optional function to validate labels for hallucinations.
                          Should take (label, code) and return list of issues found.
        """
        # Read-only store and index for retrieval
        self.store = SQLiteStore(
            db_path=db_path,
            schema=config.schema,
            store_spec=config.store_spec,
        )
        
        # Use NoOpVectorIndex if RAG is disabled
        if not config.enable_rag:
            self.index: Union[FaissVectorIndex, NoOpVectorIndex] = NoOpVectorIndex()
        else:
            self.index = FaissVectorIndex(index_dir=index_dir)
            self.index.load()
        
        self.config = config
        self.context = context
        self.validation_fn = validation_fn
        
        # Example formatter
        ex_fmt = config.examples_formatter or default_examples_formatter
        
        # Create the three agents
        self.labeler_a = LabelerAgent(
            agent_id=agent_a_id,
            schema=config.schema,
            prompt=config.labeler_a_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env="LABELER_A_MODEL",
            temperature=temperature,
            examples_formatter=ex_fmt,
        )
        
        self.labeler_b = LabelerAgent(
            agent_id=agent_b_id,
            schema=config.schema,
            prompt=config.labeler_b_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env="LABELER_B_MODEL",
            temperature=temperature,
            examples_formatter=ex_fmt,
        )
        
        self.adjudicator = AdjudicatorAgent(
            agent_id=adjudicator_id,
            schema=config.schema,
            prompt=config.adjudicator_prompt,
            patterns=config.patterns,
            store=self.store,
            index=self.index,
            model_env="ADJUDICATOR_MODEL",
            temperature=temperature,
            examples_formatter=ex_fmt,
        )
        
        # Agreement logic
        self._eq = config.label_equality_fn or (lambda x, y: x.model_dump() == y.model_dump())
    
    async def _label_async(
        self,
        agent: LabelerAgent,
        code: str,
        doc_id: int,
        k: int,
    ) -> tuple[BaseModel, str]:
        """Async wrapper for agent labeling.
        
        This allows us to run multiple labelers concurrently.
        Returns tuple of (label, prompt_text).
        """
        # Run the synchronous label method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.label, code, doc_id, k)
    
    async def label_document_async(
        self,
        code: str,
        source: Optional[str],
        doc_id: int,
    ) -> Dict[str, Any]:
        """Label a document asynchronously, parallelizing Labeler A and B.
        
        Args:
            code: Document content to label
            source: Optional source identifier
            doc_id: Document ID (already assigned by storage actor)
            
        Returns:
            Dictionary containing:
                - doc_id: Document ID
                - source: Source identifier
                - label_a: Label from Labeler A (as dict)
                - label_b: Label from Labeler B (as dict)
                - final_label: Final label (as dict)
                - decided_by: Decision method
                - agent_a_id: Labeler A identifier
                - agent_b_id: Labeler B identifier
                - prompts: Dictionary with labeler_a, labeler_b, adjudicator prompts
        """
        # Parallelize Labeler A and B using asyncio.gather (now returns tuples)
        try:
            results_a, results_b = await asyncio.gather(
                self._label_async(self.labeler_a, code, doc_id, self.config.examples_k),
                self._label_async(self.labeler_b, code, doc_id, self.config.examples_k),
            )
            label_a, prompt_a = results_a
            label_b, prompt_b = results_b
        except Exception as e:
            logger.error(f"Error during labeling for doc_id={doc_id}: {e}")
            # Return error result
            return {
                "doc_id": doc_id,
                "source": source,
                "error": str(e),
                "label_a": None,
                "label_b": None,
                "final_label": None,
                "decided_by": "error",
                "agent_a_id": self.labeler_a.agent_id,
                "agent_b_id": self.labeler_b.agent_id,
                "prompts": {"labeler_a": "", "labeler_b": "", "adjudicator": ""},
            }
        
        # Agreement or adjudication
        prompt_adjudicator = ""
        try:
            if self._eq(label_a, label_b):
                final_label = label_a
                decided_by = "agreement"
            else:
                # Adjudication is synchronous (single call, now returns tuple)
                final_label, prompt_adjudicator = self.adjudicator.decide(
                    code, label_a, label_b, doc_id, self.config.examples_k
                )
                decided_by = f"adjudicator:{self.adjudicator.agent_id}"
        except Exception as e:
            logger.error(f"Error during adjudication for doc_id={doc_id}: {e}")
            # Use label_a as fallback
            final_label = label_a
            decided_by = "error_fallback_to_a"
        
        # Validate for hallucinations if validation function is provided
        hallucination_issues = []
        if self.validation_fn:
            try:
                issues = self.validation_fn(final_label, code)
                if issues:
                    hallucination_issues = issues
                    logger.warning(
                        f"Hallucination detected in doc_id={doc_id}: {', '.join(issues)}"
                    )
            except Exception as e:
                logger.error(f"Validation function failed for doc_id={doc_id}: {e}")
        
        return {
            "doc_id": doc_id,
            "source": source,
            "label_a": label_a,
            "label_b": label_b,
            "final_label": final_label,
            "decided_by": decided_by,
            "agent_a_id": self.labeler_a.agent_id,
            "agent_b_id": self.labeler_b.agent_id,
            "hallucination_issues": hallucination_issues,
            "prompts": {
                "labeler_a": prompt_a,
                "labeler_b": prompt_b,
                "adjudicator": prompt_adjudicator,
            },
        }


