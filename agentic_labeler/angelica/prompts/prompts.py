from __future__ import annotations

"""Prompt helper utilities.

- `default_examples_formatter()` converts retrieved past labeled documents into a short,
  readable block that can be injected into LLM prompts.
- `schema_as_json()` exports the Pydantic JSON schema to help the LLM obey the schema.
"""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel


def default_examples_formatter(examples: List[Dict[str, Any]], max_chars: int = 900) -> str:
    """Format retrieval examples as a single prompt string.

    Parameters
    - examples: list of records returned from SQLiteStore.fetch_examples_for_doc_ids()
    - max_chars: maximum characters of the document snippet per example

    Returns
    - A readable text block containing a snippet + the final label JSON.
    """
    if not examples:
        return "No similar past labeled examples were found."

    blocks: List[str] = []
    for ex in examples[:5]:
        snippet = ex.get("text", "")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."

        label_json = ex.get("label_json") or "{}"
        try:
            label_obj = json.loads(label_json)
        except Exception:
            label_obj = {"raw": label_json}

        blocks.append(
            f"Example doc_id={ex.get('doc_id')}\n"
            f"Document snippet:\n{snippet}\n\n"
            f"Final label:\n{json.dumps(label_obj, indent=2, ensure_ascii=False)}"
        )

    return "\n\n".join(blocks)


def schema_as_json(schema: Type[BaseModel]) -> str:
    """Return the Pydantic model JSON schema as a pretty string."""
    return json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)
