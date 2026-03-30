"""Agent analysis
"""

from __future__ import annotations

from pathlib import Path



from tangent.agent_analysis.detect_agents.detect_agents import DetectAgents
from tangent.agent_analysis.detect_frameworks.detect_framework import DetectFramework
from tangent.agent_analysis.detect_tests.detect_agent_tests import DetectAgentTests
from tangent.agent_analysis.detect_tools.detect_tools import DetectTools
from tangent.code_analysis.code_analysis import PythonAnalysis
from tangent.agent_analysis.model.models import Application
from tangent.utils import constants
from tangent.utils.pretty import RichLog


class Analyzer:
    def __init__(self, repo: Path,
    out: Path,
    eager_analysis: bool = False,
    caller_hops: int = 1,
    analysis_json_path: Path | None = None,
    backend: str = "scalpel"):
        self.repo = repo
        self.out = out
        self.calling_depth = caller_hops
        self.analysis_json_path = analysis_json_path
        self.analysis =  self.analyze_repo(repo=repo, out=out,
                                           caller_hops=caller_hops,
                                           backend=backend,
                                           analysis_json_path=analysis_json_path,
                                           eager_analysis=eager_analysis)

    @staticmethod
    def analyze_repo(repo: Path,
                     out: Path,
                     caller_hops: int,
                     analysis_json_path: Path | None,
                     backend: str,
                     eager_analysis: bool = False
                     ) -> PythonAnalysis:
        """Run CLDK analysis and write `analysis.json`.

        Args:
            eager_analysis:
            repo: Path to the repository to analyze
            out: Path to write analysis.json
            caller_hops: Depth of call tree traversal backward from agents (default: 1)
            backend: CLDK backend to use ("scalpel" or "codeql")

        Returns PythonAnalysis object.
        """

        repo = repo.resolve()
        name = repo.name
        if analysis_json_path is None:
            out = out.resolve().joinpath(name).joinpath(constants.CODE_ANALYSIS_FILENAME)
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = analysis_json_path

        code_analysis = PythonAnalysis(
            project_dir=str(repo),
            analysis_backend_path=None,
            analysis_json_path=str(out),
            eager_analysis=eager_analysis,
            backend=backend,
        )
        return code_analysis

    def build_agent_analysis(self):
        # Step 1: Detect Framework
        frameworks = DetectFramework(analysis=self.analysis).identify_frameworks()

        # Step 2: Detect tools
        tools = DetectTools(analysis=self.analysis).detect_tools(frameworks=frameworks)

        # Step 3: Detect agents
        agents = DetectAgents(analysis=self.analysis,
                              calling_depth=self.calling_depth).detect_agents(frameworks=frameworks)

        # Step 4: Detect tests
        tests = DetectAgentTests(analysis=self.analysis,
                                 frameworks=frameworks,
                                 agents=agents).detect_agent_tests()

        # Step 5: Aggregate tools from agent.tools into the top-level tools list.
        # DetectTools (Step 2) finds decorator-based and bind_tools() call-site tools.
        # DetectAgents (Step 3) finds tools referenced in Agent(tools=[...]) constructor
        # patterns and stores them on each agent.  Merge both sources so that
        # Application.tools is the complete, deduplicated set of all tools found.
        seen_tool_ids: set = {
            (t.tool_binding_module, t.tool_binding_method, t.tool_name)
            for t in tools
        }
        for agent in agents:
            for agent_tool in agent.tools:
                tool_id = (
                    agent_tool.tool_binding_module,
                    agent_tool.tool_binding_method,
                    agent_tool.tool_name,
                )
                if tool_id not in seen_tool_ids:
                    seen_tool_ids.add(tool_id)
                    tools.append(agent_tool)

        # Step 6: Create the JSON
        agentic_solution = Application(name=self.repo.name,
                                      framework=frameworks,
                                      tools=tools,
                                      agents=agents,
                                      tests=tests)

        out = (self.out.joinpath(self.repo.name).
               joinpath(constants.AGENT_ANALYSIS_FILENAME))
        out.parent.mkdir(parents=True, exist_ok=True)
        RichLog.info("Agent analysis written to: {}".format(out))
        out.write_text(agentic_solution.model_dump_json(indent=2))
        return agentic_solution