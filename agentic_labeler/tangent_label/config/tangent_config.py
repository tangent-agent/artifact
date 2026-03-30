"""
Tangent labeling configuration for Python test modules in AI agent/tool frameworks.

Labels each test module (Python file) from the tangent JSON report with a single
boolean: is_agent_or_tool_related_test.

JSON report structure expected in ctx.extras["report_path"]:
{
  "summary": {...},
  "projects": [
    {
      "project_name": "...",
      "framework": "...",
      "num_agents": N,
      "num_tools": N,
      "test_modules": [
        {
          "module_path": "tests/test_foo.py",
          "file_path": "/absolute/path/to/tests/test_foo.py",
          "is_test_module": true,
          "test_functions": ["test_a", "test_b", ...]
        }
      ]
    }
  ]
}

Usage:
    from tangent_label.config.tangent_config import CONFIG
    ctx = LabelingContext(extras={"report_path": "/path/to/report.json"})
"""

import json
import os
from typing import Optional

from pydantic import BaseModel, Field

from angelica.models.config import (
    AgenticConfig,
    BuiltDocument,
    LabelingContext,
    LabelingUnit,
    PromptSpec,
    StoreSpec,
)


# ---------------------------------------------------------------------------
# Label schema — single boolean label
# ---------------------------------------------------------------------------

class TestLabel(BaseModel):
    is_agent_or_tool_related_test: bool = Field(
        description=(
            "True ONLY if the test module is DIRECTLY testing AI agent or tool functionality. "
            "This means the test must explicitly instantiate, configure, or invoke agents/tools. "
            "Be CONSERVATIVE: infrastructure tests, utility tests, config tests, and generic "
            "framework tests should be False. Only mark True if you see clear evidence of "
            "agent/tool behavior being tested (e.g., agent.run(), tool.invoke(), agent responses)."
        )
    )
    reasoning: str = Field(
        description=(
            "Brief explanation (2-4 sentences) of why you marked this test as agent/tool related or not. "
            "Cite specific evidence from the code: imports, class names, method calls, or test assertions. "
            "If marking False, explain what the test is actually testing instead (e.g., 'config serialization', "
            "'adapter initialization', 'utility function')."
        )
    )


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------

PATTERNS = """
⚠️ BE CONSERVATIVE: Only mark as agent/tool related if the test DIRECTLY tests agent or tool behavior.

REQUIRED signals for True (must have at least one):
1. Test instantiates an agent class (e.g., Agent(), AssistantAgent(), ConversableAgent())
2. Test calls agent methods (e.g., agent.run(), agent.chat(), agent.generate_reply())
3. Test instantiates or decorates a tool (e.g., @tool, BaseTool(), FunctionTool())
4. Test calls tool methods or verifies tool execution
5. Test verifies agent-tool interaction (agent calls tool, processes tool result)
6. Test verifies agent state/memory/conversation history
7. Test verifies multi-agent communication or orchestration

NOT sufficient alone (these are infrastructure/framework tests):
- Just importing agent frameworks (without using them)
- Testing configuration/settings classes
- Testing utility functions or helpers
- Testing serialization/deserialization
- Testing adapters or connectors (unless they test agent behavior)
- Testing mock/fixture setup code
- Testing error handling or validation (unless for agent/tool errors)
- Testing HTTP clients or API wrappers (unless for agent/tool APIs)

Examples of False:
- test_config_serialization() — just tests config, not agent behavior
- test_adapter_initialization() — just tests adapter setup, not agent usage
- test_session_creation() — just tests session management, not agent behavior
- test_error_handling() — generic error handling, not agent-specific

Examples of True:
- test_agent_generates_response() — directly tests agent behavior
- test_tool_execution() — directly tests tool behavior
- test_agent_calls_tool() — tests agent-tool interaction
- test_multi_agent_conversation() — tests agent collaboration
"""


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

LABELER_SYSTEM = """Role: Python Test Architect specializing in AI agent/tool testing.

⚠️ CRITICAL INSTRUCTIONS ⚠️
1. Analyze ONLY the provided test code below
2. Base your analysis ONLY on what is EXPLICITLY VISIBLE in the test code
3. Do NOT infer or assume the existence of methods, annotations, or features not shown
4. Be CONSERVATIVE: when in doubt, mark as False
5. Only mark True if you see DIRECT testing of agent or tool behavior (not just imports or setup)

Output JSON Schema:
{schema_json}

Return a JSON object matching the schema exactly.
"""

LABELER_HUMAN = """Analyze this Python test module:

{code}

⚠️ CRITICAL REMINDERS:
1. Only mark is_agent_or_tool_related_test=True if you see DIRECT testing of agent/tool behavior
2. Infrastructure tests, config tests, adapter tests, and utility tests should be False
3. Just importing agent frameworks is NOT sufficient — the test must USE agents/tools
4. Be CONSERVATIVE: when uncertain, mark as False
"""

ADJ_SYSTEM = """Role: Adjudicator for AI agent/tool test classification.

⚠️ CRITICAL INSTRUCTIONS ⚠️
1. Analyze ONLY the provided test code below
2. Base your decision ONLY on what is EXPLICITLY VISIBLE in the test code
3. Do NOT infer or assume the existence of methods, annotations, or features not shown
4. Be CONSERVATIVE: prefer False when uncertain
5. Only mark True if you see DIRECT agent/tool behavior testing (not just setup/config)

Output JSON Schema:
{schema_json}
"""

ADJ_HUMAN = """Test code:
{code}

Labeler A's analysis:
{a}

Labeler B's analysis:
{b}

Now review similar examples to validate your decision:
{examples}

⚠️ CRITICAL REMINDERS:
1. Make your final decision based on the test code above
2. Use examples only to validate, NOT to add features
3. Be CONSERVATIVE: only mark True if you see DIRECT agent/tool behavior testing
4. Infrastructure/config/adapter tests should be False
5. When uncertain, prefer False
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPORT_CACHE_KEY = "_tangent_report_cache"


def _load_report(ctx: LabelingContext) -> dict:
    """Load and cache the JSON report from ctx.extras['report_path']."""
    cached = ctx.get_cache(_REPORT_CACHE_KEY)
    if cached is not None:
        return cached

    report_path = ctx.extras.get("report_path")
    if not report_path:
        raise RuntimeError(
            "ctx.extras['report_path'] must be set to the path of the tangent JSON report."
        )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    ctx.set_cache(_REPORT_CACHE_KEY, report)
    return report


def _read_file_safe(file_path: str) -> str:
    """Read a file, returning an error message if it cannot be read."""
    if not file_path:
        return "[No file path provided]"
    if not os.path.isfile(file_path):
        return f"[File not found: {file_path}]"
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file {file_path}: {e}]"


def careful_examples_formatter(examples: list[dict]) -> str:
    """Format retrieved examples with clear warnings to prevent hallucinations."""
    if not examples:
        return "No reference examples available."

    lines = [
        "=" * 60,
        "REFERENCE EXAMPLES (for validation only — DO NOT use to add features)",
        "=" * 60,
        "",
        "These are DIFFERENT test modules. Use them only to validate your analysis,",
        "NOT to add features or imports to the current test.",
        "",
    ]

    for i, ex in enumerate(examples, 1):
        lines.append(f"--- Reference Example {i} ---")
        lines.append(f"Source: {ex.get('source', 'unknown')}")
        lines.append(f"is_agent_or_tool_related_test: {ex.get('is_agent_or_tool_related_test', 'unknown')}")
        code = ex.get("code", "")
        if code:
            snippet = code[:300].strip()
            if len(code) > 300:
                snippet += "..."
            lines.append(f"Code snippet:\n{snippet}")
        lines.append("")

    lines += [
        "=" * 60,
        "REMINDER: Analyze the CURRENT test module, not these examples!",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit enumerator — reads from the JSON report
# ---------------------------------------------------------------------------

def unit_enumerator(ctx: LabelingContext):
    """
    Enumerate one LabelingUnit per test module in the tangent JSON report.

    Expects ctx.extras["report_path"] to point to the JSON report file.

    Each unit's unit_id is a JSON string carrying:
      {
        "project_name": "...",
        "framework": "...",
        "num_agents": N,
        "num_tools": N,
        "module_path": "tests/test_foo.py",
        "file_path": "/absolute/path/...",
        "test_functions": ["test_a", ...]
      }
    """
    report = _load_report(ctx)

    for project in report.get("projects", []):
        project_name = project.get("project_name", "unknown")

        for module in project.get("test_modules", []):
            if not module.get("is_test_module", True):
                continue
            module_path = module.get("module_path", "")
            file_path = module.get("file_path", "")

            unit_id = json.dumps(
                {
                    "project_name": project_name,
                    "file_path": file_path,
                },
                ensure_ascii=False,
            )

            yield LabelingUnit(
                unit_type="test_module",
                unit_id=unit_id,
                source=file_path,
            )


# ---------------------------------------------------------------------------
# Unit resolver — builds the LLM prompt from the file content
# ---------------------------------------------------------------------------

def unit_resolver(unit: LabelingUnit, ctx: LabelingContext) -> BuiltDocument:
    """
    Build a BuiltDocument for a single test module.

    The LLM sees only the raw file content — nothing else.
    """
    meta = json.loads(unit.unit_id)

    file_path = meta["file_path"]

    raw_code = _read_file_safe(file_path)

    return BuiltDocument(
        content=raw_code,
        index_text=raw_code,
        metadata={
            "project_name": meta["project_name"],
            "file_path": file_path,
        },
    )


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CONFIG = AgenticConfig(
    schema=TestLabel,
    patterns=PATTERNS,
    labeler_a_prompt=PromptSpec(LABELER_SYSTEM, LABELER_HUMAN),
    labeler_b_prompt=PromptSpec(LABELER_SYSTEM, LABELER_HUMAN),
    adjudicator_prompt=PromptSpec(ADJ_SYSTEM, ADJ_HUMAN),
    examples_k=2,
    examples_formatter=careful_examples_formatter,
    store_spec=StoreSpec(
        index_fields=(
            "is_agent_or_tool_related_test",
        )
    ),
    unit_enumerator=unit_enumerator,
    unit_resolver=unit_resolver,
    enable_rag=False,
)
