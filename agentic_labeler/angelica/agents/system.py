from __future__ import annotations

"""High-level orchestration.

This is the main library API most callers use:

- Store document
- Run labeler A and labeler B
- If disagree, run adjudicator
- Store all labels (per-agent + final)
- Update vector index with the new document so it can be retrieved in the future (optional)

Two operating modes are supported:

1) Document mode (file/text-based)
   - Caller provides raw text (`code`)
   - Optional `AgenticConfig.document_builder` can inject derived signals and control
     what gets embedded vs what the LLM sees.

2) Unit mode (method/class-based, analysis-driven)
   - Caller provides a `LabelingUnit` (e.g., FQCN or "FQCN#methodSignature")
   - `AgenticConfig.unit_resolver` builds the BuiltDocument using any analysis object
     stored in `LabelingContext.analysis`.

RAG can be disabled by setting `AgenticConfig.enable_rag = False`.
"""

from dataclasses import dataclass
from typing import Optional, Union

from pydantic import BaseModel

from angelica.agents.agents import LabelerAgent, AdjudicatorAgent
from angelica.models.config import AgenticConfig, LabelingContext, BuiltDocument, LabelingUnit
from angelica.prompts.prompts import default_examples_formatter
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.noop_index import NoOpVectorIndex
from angelica.storage.sqlite.store_sqlite import SQLiteStore


@dataclass
class LabelingResult:
    """Return value from AgenticLabelingSystem.label_document()/label_unit()."""

    doc_id: int
    source: Optional[str]
    # label_a: BaseModel
    # label_b: BaseModel
    final_label: BaseModel
    decided_by: str
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


class AgenticLabelingSystem:
    """Top-level orchestrator for agentic labeling."""

    def __init__(
        self,
        store: SQLiteStore,
        index: Union[FaissVectorIndex, NoOpVectorIndex],
        config: AgenticConfig,
        context: LabelingContext | None = None,
        agent_a_id: str = "labeler_A",
        agent_b_id: str = "labeler_B",
        adjudicator_id: str = "adjudicator_1",
        temperature: float = 0.1,
    ):
        self.store = store
        self.config = config
        self.context = context or LabelingContext()
        
        # Use NoOpVectorIndex if RAG is disabled
        if not config.enable_rag:
            self.index = NoOpVectorIndex()
        else:
            self.index = index
            # Ensure index is ready before we start retrieving examples.
            self.index.load()

        # Example formatter can be overridden by the user config.
        ex_fmt = config.examples_formatter or default_examples_formatter

        # Instantiate the three agents.
        self.labeler_a = LabelerAgent(
            agent_id=agent_a_id,
            schema=config.schema,
            prompt=config.labeler_a_prompt,
            patterns=config.patterns,
            store=store,
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
            store=store,
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
            store=store,
            index=self.index,
            model_env="ADJUDICATOR_MODEL",
            temperature=temperature,
            examples_formatter=ex_fmt,
        )

        # Agreement logic is configurable; default is strict equality over full JSON.
        self._eq = config.label_equality_fn or (lambda x, y: x.model_dump() == y.model_dump())

        # Document builder for document mode
        self._builder = config.document_builder or _default_document_builder

        # Unit resolver for unit mode (optional)
        self._unit_resolver = config.unit_resolver

    def _label_built_document(self, built: BuiltDocument, source: str | None) -> LabelingResult:
        """Core labeling flow shared by document mode and unit mode."""
        import hashlib
        content_hash = hashlib.md5(built.content.encode()).hexdigest()[:8]
        
        print(f"\n📦 _label_built_document() called")
        print(f"   Source: {source}")
        print(f"   Content hash: [{content_hash}]")
        print(f"   Content preview (first 200 chars): {built.content[:200]}...")
        
        doc_id = self.store.add_document(built.content, source)
        print(f"   Stored as doc_id: {doc_id}")

        # Two independent labels (now returns label and prompt)
        print(f"   Calling labeler_a with content_hash=[{content_hash}], doc_id={doc_id}")
        a, prompt_a = self.labeler_a.label(built.content, doc_id, self.config.examples_k)
        self.store.save_label(doc_id, self.labeler_a.agent_id, a)

        print(f"   Calling labeler_b with content_hash=[{content_hash}], doc_id={doc_id}")
        b, prompt_b = self.labeler_b.label(built.content, doc_id, self.config.examples_k)
        self.store.save_label(doc_id, self.labeler_b.agent_id, b)

        # Agreement or adjudication
        prompt_adjudicator = ""
        if self._eq(a, b):
            final = a
            decided_by = "agreement"
        else:
            final, prompt_adjudicator = self.adjudicator.decide(built.content, a, b, doc_id, self.config.examples_k)
            decided_by = f"adjudicator:{self.adjudicator.agent_id}"

        # Persist final label
        self.store.save_final_label(doc_id, decided_by, final)

        # Index for retrieval (only if RAG is enabled)
        if self.config.enable_rag:
            self.index.add_document(doc_id, built.index_text, built.metadata)
            self.index.save()

        return LabelingResult(doc_id, source, final, decided_by, prompt_a, prompt_b, prompt_adjudicator)

    def label_document(self, code: str, source: Optional[str] = None) -> LabelingResult:
        """Label a single raw-text document (file/text-based mode)."""
        built = self._builder(code, source, self.context)
        return self._label_built_document(built, source)

    def label_unit(self, unit: LabelingUnit) -> LabelingResult:
        """Label a single unit (method/class/etc.) using `config.unit_resolver`.

        The resolver typically uses `self.context.analysis` to fetch code and signals.
        """
        if self._unit_resolver is None:
            raise RuntimeError(
                "No unit_resolver configured. Set AgenticConfig.unit_resolver to use label_unit()."
            )

        built = self._unit_resolver(unit, self.context)

        # Ensure stable metadata always includes unit identity
        md = dict(built.metadata or {})
        md.setdefault("unit_type", unit.unit_type)
        md.setdefault("unit_id", unit.unit_id)
        if unit.source is not None:
            md.setdefault("source", unit.source)

        built = BuiltDocument(content=built.content, index_text=built.index_text, metadata=md)

        # Use unit_id as source by default so DB rows are keyed to method/class identity.
        return self._label_built_document(built, unit.unit_id)
