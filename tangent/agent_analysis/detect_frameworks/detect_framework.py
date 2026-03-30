from typing import List, Dict, Iterable, Optional, Set

from tangent.code_analysis.code_analysis import PythonAnalysis
from tangent.utils.constants import AgenticFramework


class DetectFramework:
    def __init__(self, analysis: PythonAnalysis):
        self.analysis = analysis

    def identify_frameworks(self)->List[AgenticFramework]:
        """
                Detect agentic frameworks used in the codebase based on import roots.

                Returns:
                    Sorted list of detected frameworks, or [AgenticFramework.Unknown] if none found.
                """

        # Framework -> import roots we consider strong signals
        # Keep these as "package roots" (no dots) where possible.
        FRAMEWORK_IMPORT_ROOTS: Dict[AgenticFramework, List[str]] = {
            AgenticFramework.LangChain: [
                "langchain",
                "langchain_core",
                "langchain_community",
                "langchain_openai",
                "langchain_anthropic",
                "langchain_google_genai",
                "langchain_mistralai",
                "langchain_huggingface",
            ],
            AgenticFramework.LangGraph: ["langgraph"],
            AgenticFramework.CrewAI: ["crewai"],
            AgenticFramework.AutoGen: ["autogen", "autogen_core", "autogen_agentchat"],
            AgenticFramework.SemanticKernel: ["semantic_kernel"],
            AgenticFramework.LlamaIndex: [
                "llama_index",
            ],
            AgenticFramework.Haystack: [

                "haystack",
            ],
            AgenticFramework.MetaGPT: ["metagpt"],
            AgenticFramework.OpenAISwarm: ["swarm"],  # legacy Swarm repo/package
            AgenticFramework.OpenAIAgentsSDK: [
                # OpenAI Agents SDK is installed as openai-agents, imported as `agents`
                "agents",
            ],
            AgenticFramework.DSPy: ["dspy"],
            AgenticFramework.Langroid: ["langroid"],
            AgenticFramework.CAMEL: ["camel"],
            AgenticFramework.Letta: [
                "letta",
                "letta_client",
            ],
            AgenticFramework.MCP: [
                # Python's builtin MCP library (modelcontextprotocol/python-sdk)
                # Installed as `mcp`, imported as `mcp`
                "mcp",
                "mcp.server",
                "mcp.server.fastmcp",
                "mcp.server.stdio",
                "mcp.server.sse",
                "mcp.types",
                "mcp.client",
                "mcp.client.stdio",
                "mcp.client.sse",
                "mcp.shared",
            ],
            AgenticFramework.FastMCP: [
                # FastMCP is a higher-level wrapper around the MCP SDK
                "fastmcp",
            ],
            AgenticFramework.ClaudeAgentSDK: [
                "claude_agent_sdk",
                "anthropic.agents",
            ],
        }

        def iter_import_modules(imp) -> Iterable[str]:
            """
            Yield module-like strings to test.
            Adjust here if your analysis objects differ.
            """
            # e.g. "langchain.agents"
            if getattr(imp, "from_statement", None):
                yield imp.from_statement

            # e.g. ["langchain.agents", "pydantic"]
            for name in (getattr(imp, "imports", None) or []):
                yield name

        def _norm(s: Optional[str]) -> str:
            return (s or "").strip().lower()

        def _is_module_match(module: str, keyword: str) -> bool:
            """
            True if `module` equals `keyword` or is a submodule of it.
            Examples:
              module='langchain.agents', keyword='langchain' -> True
              module='langchain_core',  keyword='langchain' -> False
              module='my_langchain',    keyword='langchain' -> False
            """
            m = _norm(module)
            k = _norm(keyword)
            return m == k or m.startswith(k + ".")

        def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
            seen: Set[str] = set()
            out: List[str] = []
            for x in items:
                nx = _norm(x)
                if nx and nx not in seen:
                    out.append(nx)
                    seen.add(nx)
            return out

        imports = self.analysis.get_imports()

        detected: Set[AgenticFramework] = set()

        for imp in imports:
            candidates = _dedupe_preserve_order(iter_import_modules(imp))

            for fw, roots in FRAMEWORK_IMPORT_ROOTS.items():
                # Match if any candidate module matches any root
                if any(_is_module_match(mod, root) for mod in candidates for root in roots):
                    detected.add(fw)

        if not detected:
            return [AgenticFramework.Unknown]  # prefer enum over the string "unknown"

        return list(detected)
