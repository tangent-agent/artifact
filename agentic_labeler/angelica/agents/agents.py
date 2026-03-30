from __future__ import annotations

"""Agent wrappers for labelers and adjudicator.

These classes are thin shells around:
- Prompt templates (system + human)
- A structured-output LLM (Pydantic schema)
- Retrieval of similar past docs (FAISS) + example formatting (SQLite)

They do not store anything themselves; persistence happens in AgenticLabelingSystem.
"""

from typing import Any, Dict, Optional, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from angelica.llm_client.llm import make_structured_llm
from angelica.models.config import PromptSpec
from angelica.prompts.prompts import default_examples_formatter, schema_as_json
from angelica.storage.faiss.vector_faiss import FaissVectorIndex
from angelica.storage.faiss.noop_index import NoOpVectorIndex
from angelica.storage.sqlite.store_sqlite import SQLiteStore


class BaseAgent:
    """Base class for agents that produce a structured label.

    Subclasses:
    - LabelerAgent: labels a single document
    - AdjudicatorAgent: resolves disagreement between two labels
    """

    def __init__(
        self,
        agent_id: str,
        schema: Type[BaseModel],
        prompt: PromptSpec,
        patterns: str,
        store: SQLiteStore,
        index: Union[FaissVectorIndex, NoOpVectorIndex],
        model_env: str,
        temperature: float = 0.1,
        examples_formatter=None,
    ):
        self.agent_id = agent_id
        self.schema = schema
        self.patterns = patterns
        self.store = store
        self.index = index

        # Create a structured-output LLM bound to the provided schema.
        self.llm = make_structured_llm(model_env=model_env, schema=schema, temperature=temperature)

        # How retrieval examples get turned into a prompt-ready string.
        self.examples_formatter = examples_formatter or default_examples_formatter

        # Prompt templates provided by the user via config.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt.system_template),
                ("human", prompt.human_template),
            ]
        )

        # JSON schema string is included in prompts via {schema_json}.
        self._schema_json = schema_as_json(schema)

    def _retrieve_examples(self, code: str, current_doc_id: Optional[int], k: int) -> str:
        """Retrieve and format similar past examples for in-context learning.
        
        Returns empty string if index is NoOpVectorIndex (RAG disabled).
        """
        # If RAG is disabled (NoOpVectorIndex), return empty examples
        if isinstance(self.index, NoOpVectorIndex):
            # print(f"🔒 RAG DISABLED for {self.agent_id} - returning empty examples")
            return ""
        
        # print(f"🔍 RAG ENABLED for {self.agent_id} - retrieving {k} examples")
        doc_ids = self.index.similarity_doc_ids(code, k=k, exclude_doc_id=current_doc_id)
        # print(f"   Retrieved doc_ids: {doc_ids}")
        ex_df = self.store.fetch_examples_for_doc_ids(doc_ids)
        examples = ex_df.to_dict(orient="records") if not ex_df.empty else []
        # print(f"   Found {len(examples)} examples")
        # if examples:
        #     print(f"   Example sources: {[ex.get('source', 'unknown') for ex in examples]}")
        return self.examples_formatter(examples)

    def _invoke(self, vars: Dict[str, Any]) -> tuple[BaseModel, str]:
        """Run Prompt -> LLM and return the parsed Pydantic object and the formatted prompt.
        
        Returns:
            Tuple of (result, prompt_text) where prompt_text is the formatted prompt sent to LLM
        """
        # Format the prompt to get the actual text that will be sent to LLM
        formatted_prompt = self.prompt.format_messages(**vars)
        prompt_text = "\n\n".join([f"[{msg.type.upper()}]\n{msg.content}" for msg in formatted_prompt])
        
        chain = self.prompt | self.llm
        result = chain.invoke(vars)
        return result, prompt_text  # type: ignore[return-value]


class LabelerAgent(BaseAgent):
    """Independent labeler agent."""

    def label(self, code: str, current_doc_id: Optional[int], k: int) -> tuple[BaseModel, str]:
        """Label a document and return the label along with the prompt used.
        
        Returns:
            Tuple of (label, prompt_text)
        """
        import hashlib
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        # print(f"\n🏷️  {self.agent_id} labeling doc_id={current_doc_id}, code_hash=[{code_hash}]")
        # print(f"   Code preview (first 200 chars): {code[:200]}...")
        
        examples = self._retrieve_examples(code, current_doc_id, k)
        
        result = self._invoke(
            {
                "agent_id": self.agent_id,
                "patterns": self.patterns,
                "schema_json": self._schema_json,
                "code": code,
                "examples": examples,
            }
        )
        # print(f"✅ {self.agent_id} completed labeling for code_hash=[{code_hash}]\n")
        return result


class AdjudicatorAgent(BaseAgent):
    """Adjudicator that resolves disagreement between two labels."""

    def decide(self, code: str, a: BaseModel, b: BaseModel, current_doc_id: Optional[int], k: int) -> tuple[BaseModel, str]:
        """Decide between two labels and return the decision along with the prompt used.
        
        Returns:
            Tuple of (final_label, prompt_text)
        """
        examples = self._retrieve_examples(code, current_doc_id, k)
        return self._invoke(
            {
                "agent_id": self.agent_id,
                "patterns": self.patterns,
                "schema_json": self._schema_json,
                "code": code,
                "a": a.model_dump(),
                "b": b.model_dump(),
                "examples": examples,
            }
        )
