"""Ray orchestrator for coordinating parallel labeling workflow.

This module provides the main orchestrator that coordinates the entire parallel
labeling workflow, managing worker pools and batch processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for parallel processing. "
        "Install it with: pip install 'angelica[parallel]' or pip install ray"
    )

from pydantic import BaseModel
from tqdm import tqdm

from angelica.models.config import AgenticConfig, LabelingContext, LabelingUnit, BuiltDocument
from angelica.parallel.rate_limiter import RateLimiter
from angelica.parallel.ray_storage import RayStorageActor
from angelica.parallel.ray_worker import RayLabelingWorker

logger = logging.getLogger(__name__)


@dataclass
class ParallelLabelingResult:
    """Result from parallel labeling operation."""
    
    doc_id: int
    source: Optional[str]
    # label_a: BaseModel
    # label_b: BaseModel
    final_label: BaseModel
    decided_by: str
    hallucination_issues: List[str]
    prompt_a: str = ""
    prompt_b: str = ""
    prompt_adjudicator: str = ""


def _default_document_builder(code: str, source: str | None, ctx: LabelingContext) -> BuiltDocument:
    """Default builder: preserves legacy behavior."""
    return BuiltDocument(
        content=code,
        index_text=code,
        metadata={"source": source or ""},
    )


class RayLabelingOrchestrator:
    """Orchestrator for parallel document labeling using Ray.
    
    This class coordinates the entire parallel workflow:
    1. Initialize Ray and create storage actor + worker pool
    2. Distribute documents to workers in batches
    3. Collect results and persist to storage
    4. Handle rate limiting if configured
    
    Args:
        db_path: Path to SQLite database
        index_dir: Directory for FAISS index
        config: Agentic labeling configuration
        context: Labeling context
        agent_a_id: Identifier for Labeler A
        agent_b_id: Identifier for Labeler B
        adjudicator_id: Identifier for adjudicator
        temperature: LLM temperature
        num_workers: Number of parallel workers
        batch_size: Batch size for FAISS index updates
        rate_limit_rpm: Rate limit in requests per minute (None to disable)
    
    Example:
        >>> orchestrator = RayLabelingOrchestrator(
        ...     db_path="labels.db",
        ...     index_dir="vector_index",
        ...     config=config,
        ...     context=context,
        ...     num_workers=4
        ... )
        >>> results = orchestrator.label_documents_parallel(documents)
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
        num_workers: int = 4,
        batch_size: int = 10,
        rate_limit_rpm: Optional[int] = None,
        validation_fn: Optional[Callable[[BaseModel, str], List[str]]] = None,
    ):
        """Initialize the orchestrator.
        
        Args:
            validation_fn: Optional function to validate labels for hallucinations.
                          Should take (label, code) and return list of issues found.
        """
        self.db_path = db_path
        self.index_dir = index_dir
        self.config = config
        self.context = context
        self.agent_a_id = agent_a_id
        self.agent_b_id = agent_b_id
        self.adjudicator_id = adjudicator_id
        self.temperature = temperature
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.validation_fn = validation_fn
        
        # Rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_rpm)
        
        # Document builder
        self._builder = config.document_builder or _default_document_builder
        
        # Unit resolver for unit mode
        self._unit_resolver = config.unit_resolver
        
        # Ray actors (initialized lazily)
        self._storage_actor: Optional[Any] = None
        self._workers: List[Any] = []
        self._ray_initialized = False
    
    def _ensure_ray_initialized(self) -> None:
        """Initialize Ray if not already initialized."""
        if not self._ray_initialized:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            self._ray_initialized = True
    
    def _create_storage_actor(self) -> Any:
        """Create the storage actor."""
        self._ensure_ray_initialized()
        
        return RayStorageActor.remote(
            db_path=self.db_path,
            index_dir=self.index_dir,
            schema=self.config.schema,
            store_spec=self.config.store_spec,
            batch_size=self.batch_size,
            enable_rag=self.config.enable_rag,
        )
    
    def _create_workers(self) -> List[Any]:
        """Create the worker pool."""
        self._ensure_ray_initialized()
        
        workers = []
        for _ in range(self.num_workers):
            worker = RayLabelingWorker.remote(
                db_path=self.db_path,
                index_dir=self.index_dir,
                config=self.config,
                context=self.context,
                agent_a_id=self.agent_a_id,
                agent_b_id=self.agent_b_id,
                adjudicator_id=self.adjudicator_id,
                temperature=self.temperature,
                validation_fn=self.validation_fn,
            )
            workers.append(worker)
        
        return workers
    
    def _initialize_actors(self) -> None:
        """Initialize storage actor and worker pool."""
        if self._storage_actor is None:
            self._storage_actor = self._create_storage_actor()
        
        if not self._workers:
            self._workers = self._create_workers()
    
    def label_documents_parallel(
        self,
        documents: List[tuple[str, Optional[str]]],
        show_progress: bool = True,
    ) -> List[ParallelLabelingResult]:
        """Label multiple documents in parallel.
        
        Args:
            documents: List of (code, source) tuples
            show_progress: Whether to show progress bar
            
        Returns:
            List of labeling results
        """
        self._initialize_actors()
        
        results_dict: Dict[int, ParallelLabelingResult] = {}  # index -> result
        pending_tasks: Dict[Any, tuple[int, str, Dict[str, Any]]] = {}  # task_ref -> (index, index_text, metadata)
        worker_idx = 0
        
        # Progress bar
        pbar = tqdm(total=len(documents), desc="Labeling documents") if show_progress else None
        
        try:
            # Process documents
            for idx, (code, source) in enumerate(documents):
                # Rate limiting
                self.rate_limiter.acquire()
                
                # Build document
                built = self._builder(code, source, self.context)
                
                # Add to storage and get doc_id
                doc_id_ref = self._storage_actor.add_document.remote(built.content, source)
                doc_id = ray.get(doc_id_ref)
                
                # Assign to worker (round-robin)
                worker = self._workers[worker_idx]
                worker_idx = (worker_idx + 1) % self.num_workers
                
                # Submit labeling task
                task_ref = worker.label_document_async.remote(built.content, source, doc_id)
                pending_tasks[task_ref] = (idx, built.index_text, built.metadata)
                
                # Process completed tasks
                while len(pending_tasks) >= self.num_workers * 2:
                    ready_refs, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
                    for ref in ready_refs:
                        result = ray.get(ref)
                        idx, index_text, metadata = pending_tasks.pop(ref)
                        
                        # Handle results with errors - still add to results_dict
                        if isinstance(result, dict) and result.get("error"):
                            print(f"⚠️  Error in result {idx}: {result['error']}")
                            # Create a placeholder result for the error
                            error_result = ParallelLabelingResult(
                                doc_id=result.get("doc_id", -1),
                                source=result.get("source"),
                                final_label=None,  # type: ignore
                                decided_by="error",
                                hallucination_issues=[],
                                prompt_a="",
                                prompt_b="",
                                prompt_adjudicator="",
                            )
                            results_dict[idx] = error_result
                            if pbar:
                                pbar.update(1)
                            continue
                        
                        self._process_result(result, index_text, metadata)
                        results_dict[idx] = self._to_labeling_result(result)
                        if pbar:
                            pbar.update(1)
            
            # Wait for remaining tasks
            while pending_tasks:
                ready_refs, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
                for ref in ready_refs:
                    result = ray.get(ref)
                    idx, index_text, metadata = pending_tasks.pop(ref)
                    
                    # Handle results with errors - still add to results_dict
                    if isinstance(result, dict) and result.get("error"):
                        print(f"⚠️  Error in result {idx}: {result.get('error')}")
                        # Create a placeholder result for the error
                        error_result = ParallelLabelingResult(
                            doc_id=result.get("doc_id", -1),
                            source=result.get("source"),
                            final_label=None,  # type: ignore
                            decided_by="error",
                            hallucination_issues=[],
                            prompt_a="",
                            prompt_b="",
                            prompt_adjudicator="",
                        )
                        results_dict[idx] = error_result
                        if pbar:
                            pbar.update(1)
                        continue
                    
                    self._process_result(result, index_text, metadata)
                    results_dict[idx] = self._to_labeling_result(result)
                    if pbar:
                        pbar.update(1)
            
            # Flush storage
            ray.get(self._storage_actor.flush_and_save.remote())
            
        finally:
            if pbar:
                pbar.close()
        
        # Return results in original order
        return [results_dict[i] for i in range(len(documents))]
    
    def label_units_parallel(
        self,
        units: List[LabelingUnit],
        show_progress: bool = True,
    ) -> List[ParallelLabelingResult]:
        """Label multiple units in parallel.
        
        Args:
            units: List of labeling units
            show_progress: Whether to show progress bar
            
        Returns:
            List of labeling results
        """
        if self._unit_resolver is None:
            raise RuntimeError(
                "No unit_resolver configured. Set AgenticConfig.unit_resolver to use label_units_parallel()."
            )
        
        # Resolve units to documents
        documents: List[tuple[str, Optional[str]]] = []
        for i, unit in enumerate(units):

            
            built = self._unit_resolver(unit, self.context)

            
            # Ensure metadata includes unit identity
            md = dict(built.metadata or {})
            md.setdefault("unit_type", unit.unit_type)
            md.setdefault("unit_id", unit.unit_id)
            if unit.source is not None:
                md.setdefault("source", unit.source)
            
            # Use unit_id as source
            documents.append((built.content, unit.unit_id))
        
        return self.label_documents_parallel(documents, show_progress)
    
    def _process_result(
        self,
        result: Dict[str, Any],
        index_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Process a labeling result and persist to storage.
        
        Args:
            result: Result dictionary from worker
            index_text: Text to index
            metadata: Metadata for index
        """
        doc_id = result["doc_id"]
        
        # Save per-agent labels
        ray.get(self._storage_actor.save_label.remote(
            doc_id, result["agent_a_id"], result["label_a"]
        ))
        ray.get(self._storage_actor.save_label.remote(
            doc_id, result["agent_b_id"], result["label_b"]
        ))
        
        # Save final label
        ray.get(self._storage_actor.save_final_label.remote(
            doc_id, result["decided_by"], result["final_label"]
        ))
        
        # Add to index
        ray.get(self._storage_actor.add_to_index.remote(
            doc_id, index_text, metadata
        ))
    
    def _to_labeling_result(self, result: Dict[str, Any]) -> ParallelLabelingResult:
        """Convert worker result to ParallelLabelingResult.
        
        Args:
            result: Result dictionary from worker
            
        Returns:
            ParallelLabelingResult object
        """
        prompts = result.get("prompts", {})
        return ParallelLabelingResult(
            doc_id=result["doc_id"],
            source=result["source"],
            # label_a=result["label_a"],
            # label_b=result["label_b"],
            final_label=result["final_label"],
            decided_by=result["decided_by"],
            hallucination_issues=[],
            prompt_a=prompts.get("labeler_a", ""),
            prompt_b=prompts.get("labeler_b", ""),
            prompt_adjudicator=prompts.get("adjudicator", ""),
        )
    
    def shutdown(self) -> None:
        """Shutdown Ray actors and cleanup."""
        if self._storage_actor:
            ray.kill(self._storage_actor)
            self._storage_actor = None
        
        for worker in self._workers:
            ray.kill(worker)
        self._workers.clear()
        
        if self._ray_initialized and ray.is_initialized():
            ray.shutdown()
            self._ray_initialized = False


