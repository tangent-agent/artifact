from __future__ import annotations

"""Configuration models for the agentic labeling library.

This module is intentionally framework-agnostic:
- It does NOT import any particular static-analysis library.
- You can attach any expensive analysis object (e.g., a JavaAnalysis instance) via
  `LabelingContext.analysis`, or even attach callables/caches via `LabelingContext.extras`.

Core extensibility points:
- `AgenticConfig.document_builder`: build a single BuiltDocument from raw text.
- `AgenticConfig.unit_enumerator` / `AgenticConfig.unit_resolver`: switch from file-based
  labeling to unit-based labeling (methods/classes/etc.) and build prompts from identifiers
  such as fully-qualified class names (FQCNs) or method signatures.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Type, TypeVar, Generic

from pydantic import BaseModel

A = TypeVar("A")  # analysis type (kept generic on purpose)


@dataclass
class LabelingContext(Generic[A]):
    """Shared context built once per run.

    Put expensive objects and caches here:
    - project_path: repository root or project root
    - analysis: a static analysis object (e.g., JavaAnalysis)
    - extras: free-form cache / data (always a dict)

    The library treats `analysis` as opaque; your config functions decide how to use it.
    """

    project_path: str | None = None
    analysis: A | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def get_cache(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)

    def set_cache(self, key: str, value: Any) -> None:
        self.extras[key] = value

    def require_analysis(self) -> A:
        if self.analysis is None:
            raise RuntimeError(
                "LabelingContext.analysis is None. " 
                "Build your analysis object in the runner/CLI and pass it via LabelingContext(analysis=...)."
            )
        return self.analysis


@dataclass(frozen=True)
class BuiltDocument:
    """Prepared unit that will be stored, labeled, and indexed.

    content:
        What is passed to the LLM prompt as {code}. This may include derived signals.
    index_text:
        What is embedded for retrieval. Usually keep this stable (e.g., code only).
    metadata:
        Small JSON-serializable metadata stored in the vector index (FAISS).
    """

    content: str
    index_text: str
    metadata: dict[str, Any]


# Builder signature for file/text-based labeling:
# (raw_text, source, context) -> BuiltDocument
DocumentBuilder = Callable[[str, str | None, LabelingContext[Any]], BuiltDocument]


@dataclass(frozen=True)
class LabelingUnit:
    """A single labeling target (method/class/file/etc.).

    unit_type:
        A short type label such as "file", "class", "method", or any custom value.
    unit_id:
        A stable identifier for this unit (e.g., FQCN or "FQCN#methodSignature").
    source:
        Optional human-readable source hint (e.g., a file path).
    """

    unit_type: str
    unit_id: str
    source: str | None = None


# Unit-based labeling hooks:
# - enumerator decides *what* to label (returns units)
# - resolver decides *how* to build prompts/index text for a given unit
UnitEnumerator = Callable[[LabelingContext[Any]], Iterable[LabelingUnit]]
UnitResolver = Callable[[LabelingUnit, LabelingContext[Any]], BuiltDocument]


@dataclass(frozen=True)
class PromptSpec:
    """Prompt templates used for an agent.

    Both templates are plain Python format strings.

    Variables supplied by the system:
    - {patterns}: pattern catalog / taxonomy
    - {schema_json}: JSON schema for the output model
    - {code}: document content (BuiltDocument.content)
    - {examples}: retrieved examples text block
    - {a}, {b}: labeler outputs (adjudicator only)
    - {agent_id}: agent identifier
    """

    system_template: str
    human_template: str


@dataclass(frozen=True)
class StoreSpec:
    """Storage behavior for SQLite.

    Labels are always stored as JSON in the column named by `json_column`.

    Optionally, you can store selected top-level fields as separate columns
    (useful for filtering/metrics).
    """

    json_column: str = "label_json"
    index_fields: tuple[str, ...] = ("",)  # empty string => store none


@dataclass(frozen=True)
class AgenticConfig:
    """Top-level config object.

    You supply:
    - schema: Pydantic model for structured output
    - patterns: a taxonomy / pattern catalog string
    - prompts: PromptSpec for labeler A, labeler B, adjudicator
    - examples_k: how many retrieved examples to include

    Optional hooks:
    - document_builder: customize prompt/index text for file/text-based labeling
    - unit_enumerator + unit_resolver: enable unit-based labeling (methods/classes/etc.)
    - label_equality_fn: define agreement between labelers
    - examples_formatter: customize retrieved example formatting
    - store_spec: control SQLite storage of selected fields
    - enable_rag: enable/disable vector indexing and retrieval (default: True)
    """

    schema: Type[BaseModel]
    labeler_a_prompt: PromptSpec
    labeler_b_prompt: PromptSpec
    adjudicator_prompt: PromptSpec

    patterns: str = ""
    examples_k: int = 5

    label_equality_fn: Optional[Callable[[BaseModel, BaseModel], bool]] = None
    examples_formatter: Optional[Callable[[list[dict[str, Any]]], str]] = None
    store_spec: StoreSpec = StoreSpec()

    document_builder: Optional[DocumentBuilder] = None

    unit_enumerator: Optional[UnitEnumerator] = None
    unit_resolver: Optional[UnitResolver] = None
    
    enable_rag: bool = True
