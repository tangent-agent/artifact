import os
from typing import List, Set, Tuple
import ast

from tangent.code_analysis.code_analysis import PythonAnalysis
from tangent.code_analysis.model.model import PyCallSite, PyFunction, PyModule
from tangent.agent_analysis.model.models import Agent, CallableDetails, Tool
from tangent.utils.constants import AgenticFramework, AgentType, ToolType


class DetectAgents:
    def __init__(self, analysis: PythonAnalysis, calling_depth: int):
        self.analysis = analysis
        self.calling_depth = calling_depth
    
    def _has_import(self, module_substring: str) -> bool:
        """Check if any import contains the given substring."""
        for imp in self.analysis.get_imports():
            if module_substring in imp.from_statement:
                return True
            if any(module_substring in name for name in imp.imports):
                return True
        return False
    
    def _module_has_import(self, module: PyModule, module_substring: str) -> bool:
        """Check if a specific module has imports containing the given substring."""
        if not module or not module.imports:
            return False
        for imp in module.imports:
            if module_substring in imp.from_statement:
                return True
            if any(module_substring in name for name in imp.imports):
                return True
        return False
    
    def _find_constructor_calls(self, class_names: List[str]) -> List[PyCallSite]:
        """Find all constructor calls for the given class names."""
        results = []
        for call_site in self.analysis.get_call_sites():
            if call_site.is_constructor_call and call_site.method_name in class_names:
                results.append(call_site)
        return results
    
    def _find_method_calls(self, method_names: List[str]) -> List[PyCallSite]:
        """Find all method calls with the given names."""
        results = []
        for call_site in self.analysis.get_call_sites():
            if call_site.method_name in method_names:
                results.append(call_site)
        return results
    
    def _get_function_for_callsite(self, call_site: PyCallSite) -> PyFunction | None:
        """Find the function/method containing a call site."""
        for func in self.analysis.iter_functions():
            if call_site in func.call_sites:
                return func
        return None
    
    def _get_module_for_function(self, func: PyFunction) -> PyModule | None:
        """Get the module containing a function."""
        for module in self.analysis.get_modules():
            if func in module.functions:
                return module
            for cls in module.classes:
                if func in cls.methods:
                    return module
        return None
    
    def _calculate_cyclomatic_complexity(self, func: PyFunction) -> int:
        """Calculate cyclomatic complexity for a function.
        
        Cyclomatic complexity = number of decision points + 1
        Decision points include: if, elif, for, while, except, and, or, with
        """
        module = self._get_module_for_function(func)
        if not module:
            return 1  # Default complexity
        
        try:
            from pathlib import Path
            source = Path(module.file_path).read_text(encoding='utf-8')
            tree = ast.parse(source)
            
            # Find the function node
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == func.name and node.lineno == func.start_line:
                        # Count decision points
                        complexity = 1  # Base complexity
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                                complexity += 1
                            elif isinstance(child, ast.BoolOp):
                                # Count and/or operators
                                complexity += len(child.values) - 1
                        return complexity
        except Exception:
            pass
        
        return 1  # Default if calculation fails
    
    def _calculate_ncloc(self, func: PyFunction) -> int:
        """Calculate non-comment lines of code for a function."""
        if func.start_line < 0 or func.end_line < 0:
            return 0
        
        module = self._get_module_for_function(func)
        if not module:
            return 0
        
        try:
            from pathlib import Path
            source = Path(module.file_path).read_text(encoding='utf-8')
            lines = source.split('\n')
            
            # Count non-empty, non-comment lines
            ncloc = 0
            for i in range(func.start_line - 1, min(func.end_line, len(lines))):
                line = lines[i].strip()
                if line and not line.startswith('#'):
                    ncloc += 1
            
            return ncloc
        except Exception:
            return 0
    
    def _collect_call_details(self, func: PyFunction) -> Tuple[List[CallableDetails], List[CallableDetails], List[CallableDetails]]:
        """Collect constructor, application, and library call details from a function."""
        constructor_calls = []
        application_calls = []
        library_calls = []
        
        for call_site in func.call_sites:
            # Filter out None values from argument_types
            arg_types = [t for t in (call_site.argument_types or []) if t is not None]
            
            call_detail = CallableDetails(
                method_name=call_site.method_name,
                qualified_class_name=call_site.receiver_type or "",
                qualified_module_name=call_site.qualified_module_name,
                qualified_method_signature=call_site.method_signature,
                line_number=call_site.start_line,
                column_number=call_site.start_column,
                argument_types=arg_types,
                receiver_type=call_site.receiver_type,
            )
            
            if call_site.is_constructor_call:
                constructor_calls.append(call_detail)
            elif call_site.is_application_call:
                application_calls.append(call_detail)
            elif call_site.is_library_call:
                library_calls.append(call_detail)
        
        return constructor_calls, application_calls, library_calls
    
    def _detect_agent_tools(self, func: PyFunction) -> List[Tool]:
        """Detect tools used or referenced in an agent function.
        
        This method looks for:
        1. Tool references in function arguments (e.g., tools=[...])
        2. Tool binding calls (e.g., agent.bind_tools([...]))
        3. Tool decorator usage on functions called within this agent
        
        Args:
            func: The function to analyze for tool usage
            
        Returns:
            List of Tool objects detected in this function
        """
        tools = []
        tool_names_seen = set()
        
        # Look for tool-related call sites in the function
        for call_site in func.call_sites:
            # Check for tool binding methods
            if call_site.method_name in ["bind_tools", "with_tools", "add_tool", "register_tool"]:
                # Try to extract tool names from arguments
                for arg_expr in call_site.argument_expr:
                    if arg_expr and not arg_expr.startswith('"') and not arg_expr.startswith("'"):
                        # Extract potential tool names from the expression
                        # Handle list expressions like [tool1, tool2]
                        if '[' in arg_expr and ']' in arg_expr:
                            # Extract items from list
                            list_content = arg_expr[arg_expr.find('[')+1:arg_expr.rfind(']')]
                            tool_refs = [t.strip() for t in list_content.split(',') if t.strip()]
                        else:
                            tool_refs = [arg_expr.strip()]
                        
                        for tool_ref in tool_refs:
                            if tool_ref and tool_ref not in tool_names_seen:
                                tool_names_seen.add(tool_ref)
                                tools.append(Tool(
                                    tool_name=tool_ref,
                                    tool_type=ToolType.FUNCTION_TOOL,
                                    tool_description=None,
                                    tool_binding_method=func.qualified_name,
                                    tool_binding_module=func.qualified_module_name
                                ))
            
            # Check for tool parameter in constructor calls (e.g., Agent(tools=[...]))
            if call_site.is_constructor_call:
                for i, arg in enumerate(call_site.arguments):
                    if arg.name == "tools" and i < len(call_site.argument_expr):
                        arg_expr = call_site.argument_expr[i]
                        if arg_expr and '[' in arg_expr and ']' in arg_expr:
                            list_content = arg_expr[arg_expr.find('[')+1:arg_expr.rfind(']')]
                            tool_refs = [t.strip() for t in list_content.split(',') if t.strip()]
                            
                            for tool_ref in tool_refs:
                                if tool_ref and tool_ref not in tool_names_seen:
                                    tool_names_seen.add(tool_ref)
                                    tools.append(Tool(
                                        tool_name=tool_ref,
                                        tool_type=ToolType.FUNCTION_TOOL,
                                        tool_description=None,
                                        tool_binding_method=func.qualified_name,
                                        tool_binding_module=func.qualified_module_name
                                    ))
        
        return tools
    
    def _agent_exists(self, agents: List[Agent], method_signature: str, qualified_module_name: str) -> bool:
        """Check if an agent with the same method signature and module name already exists."""
        for agent in agents:
            if agent.method_signature == method_signature and agent.qualified_module_name == qualified_module_name:
                return True
        return False

    def detect_agents(self, frameworks: List[AgenticFramework])->List[Agent]:
        factory_agents = []
        for framework in frameworks:
            match framework:
                case AgenticFramework.LangGraph:
                    factory_agents.extend(self.detect_langgraph_agent())
                case AgenticFramework.LangChain:
                    factory_agents.extend(self.detect_langchain_agent())
                case AgenticFramework.CrewAI:
                    factory_agents.extend(self.detect_crewai_agent())
                case AgenticFramework.AutoGen:
                    factory_agents.extend(self.detect_autogen_agent())
                case AgenticFramework.SemanticKernel:
                    factory_agents.extend(self.detect_semantickernel_agent())
                case AgenticFramework.LlamaIndex:
                    factory_agents.extend(self.detect_llamaindex_agent())
                case AgenticFramework.Haystack:
                    factory_agents.extend(self.detect_haystack_agent())
                case AgenticFramework.MetaGPT:
                    factory_agents.extend(self.detect_metagpt_agent())
                case AgenticFramework.Letta:
                    factory_agents.extend(self.detect_letta_agent())
                case AgenticFramework.OpenAISwarm:
                    factory_agents.extend(self.detect_swarm_agent())
                case AgenticFramework.DSPy:
                    factory_agents.extend(self.detect_dspy_agent())
                case AgenticFramework.OpenAIAgentsSDK:
                    factory_agents.extend(self.detect_openai_sdk_agent())
                case AgenticFramework.Langroid:
                    factory_agents.extend(self.detect_langroid_agent())
                case AgenticFramework.CAMEL:
                    factory_agents.extend(self.detect_camel_agent())
                case AgenticFramework.ClaudeAgentSDK:
                    factory_agents.extend(self.detect_claude_agent_sdk_agent())
                case AgenticFramework.MCP:
                    factory_agents.extend(self.detect_mcp_agent())
                case AgenticFramework.FastMCP:
                    factory_agents.extend(self.detect_fastmcp_agent())
                case _:
                    pass
        calling_agents = self.detect_calling_agents(agents=factory_agents)
        all_agents = factory_agents + calling_agents
        return all_agents
    def _extract_factory_module_names(self, factory_agent_modules: set) -> set:
        """Extract simple module names from full module paths.
        
        Args:
            factory_agent_modules: Set of full module paths
            
        Returns:
            Set of simple module names (last part of path)
        """
        factory_module_names = set()
        for mod_name in factory_agent_modules:
            if mod_name:
                # Remove .py extension and get last part (e.g., "agents" from "myapp/agents.py")
                mod_name = mod_name.replace(".py", "")
                parts = mod_name.split(os.sep)
                factory_module_names.add(parts[-1])
        return factory_module_names
    
    def _module_imports_factory(self, module, factory_module_names: set) -> bool:
        """Check if a module imports from any factory agent module.
        
        Args:
            module: PyModule to check
            factory_module_names: Set of factory module names to look for
            
        Returns:
            True if module imports from a factory module
        """
        for imp in module.imports:
            # Check from_statement (e.g., "from myapp.agents import ...")
            if imp.from_statement:
                for factory_mod in factory_module_names:
                    if factory_mod in imp.from_statement or imp.from_statement.endswith(factory_mod):
                        return True
            
            # Check imported names (e.g., "import myapp.agents")
            for imported_name in imp.imports:
                name_parts = imported_name.split('.')
                # Check last 2 parts for module name
                if len(name_parts) > 1:
                    name_parts = name_parts[-2:]
                
                for part in name_parts:
                    if part in factory_module_names:
                        return True
        
        return False
    
    def _find_called_factory_agents(self, function_or_method, agents: List[Agent],
                                    factory_agent_names: set) -> List[str]:
        """Find all factory agents called by a function or method.
        
        Args:
            function_or_method: PyFunction or PyMethod to analyze
            agents: List of all known agents
            factory_agent_names: Set of factory agent method signatures
            
        Returns:
            List of called factory agent method signatures
        """
        called_factory_agents = []
        agent_name_map = {agent.name: agent for agent in agents}
        
        for call_site in function_or_method.call_sites:
            # Method 1: Direct signature match
            if call_site.method_signature in factory_agent_names:
                called_factory_agents.append(call_site.method_signature)
                continue
            
            # Method 2: Match by method name and module
            if call_site.method_name in agent_name_map:
                agent = agent_name_map[call_site.method_name]
                if agent.qualified_module_name == call_site.qualified_module_name:
                    if agent.method_signature not in called_factory_agents:
                        called_factory_agents.append(agent.method_signature)
                    continue
            
            # Method 3: Check if any argument value matches an agent name
            for arg in call_site.arguments:
                if arg.name in agent_name_map:
                    agent = agent_name_map[arg.name]
                    if agent.method_signature not in called_factory_agents:
                        called_factory_agents.append(agent.method_signature)
        
        return called_factory_agents
    
    def _create_calling_agent(self, function_or_method, called_factory_agents: List[str],
                             agents: List[Agent], qualified_class_name: str = "") -> Agent:
        """Create a calling agent from a function or method.
        
        Args:
            function_or_method: PyFunction or PyMethod
            called_factory_agents: List of called factory agent signatures
            agents: List of all known agents
            qualified_class_name: Class name if this is a method
            
        Returns:
            Agent object representing the calling agent
        """
        # Collect call details and metrics
        constructor_calls, application_calls, library_calls = self._collect_call_details(function_or_method)
        
        # Detect tools used in this agent
        agent_tools = self._detect_agent_tools(function_or_method)
        
        ncloc = self._calculate_ncloc(function_or_method)
        complexity = self._calculate_cyclomatic_complexity(function_or_method)
        num_objects = len(constructor_calls)
        
        # Determine framework from called agents
        frameworks = {agent.framework for agent in agents
                     if agent.method_signature in called_factory_agents}
        primary_framework = list(frameworks)[0] if frameworks else AgenticFramework.LangChain
        
        return Agent(
            name=function_or_method.name,
            agent_type=AgentType.CALLER,
            framework=primary_framework,
            qualified_class_name=qualified_class_name,
            qualified_module_name=function_or_method.qualified_module_name,
            method_signature=function_or_method.qualified_name,
            tools=agent_tools,
            description=f"Calling agent that invokes: {', '.join(called_factory_agents)}",
            factory_agent_names=called_factory_agents,
            ncloc=ncloc,
            ncloc_with_helpers=ncloc,
            cyclomatic_complexity=complexity,
            cyclomatic_complexity_with_helpers=complexity,
            number_of_objects_created=num_objects,
            constructor_call_details=constructor_calls,
            application_call_details=application_calls,
            library_call_details=library_calls,
        )
    
    def detect_calling_agents(self, agents: List[Agent]) -> List[Agent]:
        """Detect calling agents - functions that invoke factory agents.
        
        A calling agent is a non-test function/method that:
        1. Is in a module that imports from a factory agent module
        2. Calls a factory agent function (directly or via arguments)
        
        Args:
            agents: List of factory agents to search for callers
            
        Returns:
            List of detected calling agents
        """
        calling_agents = []
        
        if not agents:
            return calling_agents
        
        # Build lookup structures
        factory_agent_names = {agent.method_signature for agent in agents}
        factory_agent_modules = {agent.qualified_module_name for agent in agents}
        factory_module_names = self._extract_factory_module_names(factory_agent_modules)
        
        # Scan all non-test modules
        for module in self.analysis.get_modules():
            if module.is_test:
                continue
            
            # Skip modules that don't import factory modules
            if not self._module_imports_factory(module, factory_module_names):
                continue
            
            # Check module-level functions
            for func in module.functions:
                if func.is_test:
                    continue
                
                called_factory_agents = self._find_called_factory_agents(
                    func, agents, factory_agent_names
                )
                
                if called_factory_agents:
                    # Skip if already exists
                    if self._agent_exists(calling_agents, func.qualified_name,
                                        func.qualified_module_name):
                        continue
                    
                    calling_agent = self._create_calling_agent(
                        func, called_factory_agents, agents
                    )
                    calling_agents.append(calling_agent)
            
            # Check class methods
            for cls in module.classes:
                for method in cls.methods:
                    if method.is_test:
                        continue
                    
                    called_factory_agents = self._find_called_factory_agents(
                        method, agents, factory_agent_names
                    )
                    
                    if called_factory_agents:
                        # Skip if already exists
                        if self._agent_exists(calling_agents, method.qualified_name,
                                            method.qualified_module_name):
                            continue
                        
                        calling_agent = self._create_calling_agent(
                            method, called_factory_agents, agents, cls.qualified_name
                        )
                        calling_agents.append(calling_agent)
        
        return calling_agents

    def detect_langchain_agent(self)->List[Agent]:
        """Detect LangChain agents.
        
        Primary creation patterns:
        - Common factory calls (often in langchain.agents):
          from langchain.agents import create_agent
        - Also widely seen in codebases:
          create_react_agent(llm, tools, ...)
          create_tool_calling_agent(llm, tools, ...)
          initialize_agent(tools, llm, agent=..., ...) (older style, still appears)
        - RunnablePassthrough.assign() pattern:
          agent = (
              RunnablePassthrough.assign(
                  agent_scratchpad=lambda x: format_to_platform_tool_messages(x["intermediate_steps"]),
              )
              | prompt
              | llm_with_stop
              | PlatformToolsAgentOutputParser(instance_type="base")
          )
        
        Execution patterns:
        - agent_executor = AgentExecutor(...)
        - agent_executor.invoke({"input": ...}) or agent_executor.run(...)
        
        Detection keys:
        - Imports containing langchain.agents or RunnablePassthrough
        - Calls named: create_agent, create_react_agent, create_tool_calling_agent, initialize_agent, AgentExecutor(...)
        - RunnablePassthrough.assign() calls
        """
        agents = []
        
        if not self._has_import("langchain"):
            return agents
        
        # Look for agent creation function calls
        agent_creation_calls = self._find_method_calls([
            "create_agent", "create_react_agent", "create_tool_calling_agent",
            "initialize_agent"
        ])
        
        # Look for AgentExecutor constructor calls
        executor_calls = self._find_constructor_calls(["AgentExecutor"])

        # Look for RunnablePassthrough.assign() pattern
        runnable_passthrough_calls = []
        if self._has_import("RunnablePassthrough"):
            assign_calls = self._find_method_calls(["assign"])
            # Filter to only those where receiver type contains RunnablePassthrough
            for call_site in assign_calls:
                if call_site.receiver_type and "RunnablePassthrough" in call_site.receiver_type:
                    runnable_passthrough_calls.append(call_site)

        all_calls = agent_creation_calls + executor_calls + runnable_passthrough_calls

        for call_site in all_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has langchain imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "langchain"):
                    continue

                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue

                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)

                # Detect tools used in this agent
                agent_tools = self._detect_agent_tools(func)

                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)

                # Determine description based on call pattern
                if call_site.method_name == "assign" and call_site.receiver_type and "RunnablePassthrough" in call_site.receiver_type:
                    description = f"LangChain agent created via RunnablePassthrough.assign() pattern"
                else:
                    description = f"LangChain agent created via {call_site.method_name}"
                
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.LangChain,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=agent_tools,
                    description=description,
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_langgraph_agent(self)->List[Agent]:
        """Detect LangGraph agents.
        
        Primary "agent factory" patterns:
        - LangGraph prebuilt (older):
          from langgraph.prebuilt import create_react_agent
          agent = create_react_agent(model, tools, ...)
        - Newer migration path: LangGraph's create_react_agent is deprecated in favor of langchain.agents.create_agent
        
        Execution patterns:
        - agent.invoke(...), agent.stream(...), or compiled graph app.invoke(...) if you see:
          graph = StateGraph(...)
          app = graph.compile()
          app.invoke(state) / app.stream(state)
        
        Detection keys:
        - Imports containing langgraph
        - Calls to StateGraph(...), .add_node(...), .add_edge(...), .compile()
        - Calls named create_react_agent(...) (either langgraph or langchain depending on import)
        """
        agents = []
        
        if not self._has_import("langgraph"):
            return agents
        
        # Look for StateGraph constructor calls
        state_graph_calls = self._find_constructor_calls(["StateGraph"])
        
        # Look for create_react_agent calls
        react_agent_calls = self._find_method_calls(["create_react_agent"])
        
        all_calls = state_graph_calls + react_agent_calls
        
        for call_site in all_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has langgraph imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "langgraph"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.LangGraph,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"LangGraph agent created via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_crewai_agent(self)->List[Agent]:
        """Detect CrewAI agents.
        
        Direct-in-code agent construction:
        - from crewai import Agent
        - agent = Agent(role=..., goal=..., backstory=..., tools=[...], ...)
        - CrewAI also supports YAML-defined agents (so creation may be indirect), but it's still "agent creation" semantically
        
        Orchestration (often paired with agent creation):
        - from crewai import Crew, Task
        - crew = Crew(agents=[...], tasks=[...])
        - crew.kickoff(inputs={...})
        
        Detection keys:
        - Imports containing crewai
        - Agent(...) constructor calls
        - Optional: YAML load patterns referencing agents.yaml + subsequent Crew(...).kickoff(...)
        """
        agents = []
        
        if not self._has_import("crewai"):
            return agents
        
        # Look for Agent constructor calls
        agent_calls = self._find_constructor_calls(["Agent"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has crewai imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "crewai"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.CrewAI,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="CrewAI agent created via Agent constructor",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_autogen_agent(self)->List[Agent]:
        """Detect AutoGen agents.
        
        Common "agent" constructors (classic surface API):
        - from autogen import AssistantAgent, UserProxyAgent, LLMConfig
        - assistant = AssistantAgent("assistant", llm_config=...)
        - user_proxy = UserProxyAgent("user_proxy", ...)
        - Execution: user_proxy.run(assistant, message="...").process()
        
        Newer modular packages (often in newer docs):
        - from autogen_agentchat.agents import AssistantAgent
        - from autogen_ext.models.openai import OpenAIChatCompletionClient
        - AssistantAgent(..., model_client=OpenAIChatCompletionClient(...))
        
        Detection keys:
        - Imports containing autogen, autogen_agentchat, autogen_ext
        - Constructor calls: AssistantAgent(...), UserProxyAgent(...), ConversableAgent(...)
        - Run loop calls: .run(...), .initiate_chat(...), .process()
        """
        agents = []
        
        if not (self._has_import("autogen") or self._has_import("autogen_agentchat") or self._has_import("autogen_ext")):
            return agents
        
        # Look for agent constructor calls
        agent_calls = self._find_constructor_calls([
            "AssistantAgent", "UserProxyAgent", "ConversableAgent"
        ])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has autogen imports
                module = self._get_module_for_function(func)
                if not module or not (self._module_has_import(module, "autogen") or
                                     self._module_has_import(module, "autogen_agentchat") or
                                     self._module_has_import(module, "autogen_ext")):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.AutoGen,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"AutoGen agent created via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_semantickernel_agent(self)->List[Agent]:
        """Detect Semantic Kernel agents.
        
        Agent type constructors:
        - A very common pattern is using a ChatCompletionAgent:
          from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
          agent = ChatCompletionAgent(...)
        
        Invocation patterns:
        - agent.invoke(...) / agent.invoke_stream(...) are the telltale execution entrypoints mentioned in the agent docs
        
        Detection keys:
        - Imports containing semantic_kernel.agents
        - Constructor calls to ChatCompletionAgent(...)
        - Calls to .invoke(...) / .invoke_stream(...)
        """
        agents = []
        
        if not self._has_import("semantic_kernel"):
            return agents
        
        # Look for ChatCompletionAgent constructor calls
        agent_calls = self._find_constructor_calls(["ChatCompletionAgent"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has semantic_kernel imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "semantic_kernel"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.SemanticKernel,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="Semantic Kernel agent created via ChatCompletionAgent",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_llamaindex_agent(self)->List[Agent]:
        """Detect LlamaIndex agents.
        
        "Agent" creation patterns (prebuilt agents/workflows):
        - Docs explicitly call out:
          * FunctionAgent (tool/function calling agent)
          * AgentWorkflow (multi-agent managing workflow)
        - In practice you'll see imports like:
          from llama_index.core.agent import FunctionAgent (or similar module paths depending on version)
          agent = FunctionAgent(tools=[...], llm=...)
          workflow = AgentWorkflow(...)
        
        Execution patterns:
        - agent.run(...), agent.chat(...), workflow.run(...) (varies by agent type/version)
        
        Detection keys:
        - Imports containing llama_index
        - Constructor calls containing FunctionAgent, AgentWorkflow, ReActAgent (also commonly used in examples)
        - Calls to .chat(...) / .run(...)
        """
        agents = []
        
        if not self._has_import("llama_index"):
            return agents
        
        # Look for agent constructor calls
        agent_calls = self._find_constructor_calls([
            "FunctionAgent", "AgentWorkflow", "ReActAgent"
        ])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has llama_index imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "llama_index"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.LlamaIndex,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"LlamaIndex agent created via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_haystack_agent(self)->List[Agent]:
        """Detect Haystack agents.
        
        Agent component construction (pipeline component):
        - Haystack 2.x defines an Agent component:
          from haystack.components.agents import Agent
          agent = Agent(chat_generator=..., tools=[...], system_prompt=..., exit_conditions=[...])
        - Tools can be created via:
          Tool(...), ComponentTool(...), or @tool decorator
        
        Execution patterns:
        - result = agent.run(messages=[...])
        
        Detection keys:
        - Imports containing haystack.components.agents or haystack.components.agents.Agent
        - Constructor call Agent(chat_generator=...)
        - .run(messages=...)
        - Tool wrappers: ComponentTool(component=...), Tool(...), @tool
        """
        agents = []
        
        if not self._has_import("haystack"):
            return agents
        
        # Look for Agent constructor calls
        agent_calls = self._find_constructor_calls(["Agent"])
        
        # Filter to only Haystack agents (check if import is from haystack.components.agents)
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Check if the module has haystack imports
                module = self.analysis.get_module(func.qualified_module_name)
                if module and any("haystack" in imp.from_statement for imp in module.imports):
                    # Check if agent already exists
                    if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                        continue
                    
                    # Collect call details
                    constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                    
                    # Calculate metrics
                    ncloc = self._calculate_ncloc(func)
                    complexity = self._calculate_cyclomatic_complexity(func)
                    num_objects = len(constructor_calls)
                    
                    agent = Agent(
                        name=func.name,
                        agent_type=AgentType.FACTORY,
                        framework=AgenticFramework.Haystack,
                        qualified_class_name="",
                        qualified_module_name=func.qualified_module_name,
                        method_signature=func.qualified_name,
                        tools=[],
                        description="Haystack agent created via Agent component",
                        ncloc=ncloc,
                        ncloc_with_helpers=ncloc,
                        cyclomatic_complexity=complexity,
                        cyclomatic_complexity_with_helpers=complexity,
                        number_of_objects_created=num_objects,
                        constructor_call_details=constructor_calls,
                        application_call_details=application_calls,
                        library_call_details=library_calls,
                    )
                    agents.append(agent)
        
        return agents

    def detect_metagpt_agent(self)->List[Agent]:
        """Detect MetaGPT agents.
        
        MetaGPT's "agent" concept is often expressed as a Role (and sometimes Agent) that you instantiate and run().
        
        Off-the-shelf role instantiation:
        - from metagpt.roles.product_manager import ProductManager
        - role = ProductManager()
        - result = await role.run(msg)
        
        Custom role/agent patterns (common in repos):
        - from metagpt.roles.role import Role
        - role = Role(name=..., actions=[...], ...)
        - await role.run(...)
        
        Detection keys:
        - Imports containing metagpt.roles
        - Instantiation of subclasses of Role
        - Any awaited .run(...) on a role/agent object
        """
        agents = []
        
        if not self._has_import("metagpt"):
            return agents
        
        # Look for Role constructor calls and subclasses
        role_calls = self._find_constructor_calls(["Role", "ProductManager", "Architect", "Engineer"])
        
        for call_site in role_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has metagpt imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "metagpt"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.MetaGPT,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"MetaGPT agent created via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_swarm_agent(self)->List[Agent]:
        """Detect OpenAI Swarm agents.
        
        Agent construction:
        - from swarm import Agent
        - agent = Agent(name=..., model=..., instructions=..., functions=[...])
        
        Run loop:
        - client.run(agent=agent, messages=[...], context_variables={...})
        
        Detection keys:
        - Imports containing swarm
        - Agent(...) constructor call
        - client.run(...) where an Agent instance is passed
        """
        agents = []
        
        if not self._has_import("swarm"):
            return agents
        
        # Look for Agent constructor calls
        agent_calls = self._find_constructor_calls(["Agent"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has swarm imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "swarm"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.OpenAISwarm,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="OpenAI Swarm agent created via Agent constructor",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_dspy_agent(self)->List[Agent]:
        """Detect DSPy agents.
        
        DSPy's "agent" is often a ReAct module or an agent-like DSPy Module.
        
        ReAct creation pattern:
        - import dspy
        - agent = dspy.ReAct(signature=..., tools=[...], ...)
        
        Custom agent-as-module pattern:
        - class MyAgent(dspy.Module):
            def forward(self, ...): ...
        - Instantiation: my_agent = MyAgent(...)
        
        Detection keys:
        - dspy.ReAct(...) calls
        - Class defs inheriting dspy.Module
        - Tool usage via dspy.Tool / tool signatures (often appears alongside ReAct)
        """
        agents = []
        
        if not self._has_import("dspy"):
            return agents
        
        # Look for ReAct constructor calls
        react_calls = self._find_constructor_calls(["ReAct"])
        
        for call_site in react_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has dspy imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "dspy"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.DSPy,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="DSPy agent created via ReAct",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        # TODO: Also detect dspy.Module subclasses
        
        return agents

    def detect_openai_sdk_agent(self)->List[Agent]:
        """Detect OpenAI Agents SDK agents.
        
        Core creation pattern:
        - from agents import Agent, Runner
        - agent = Agent(name="...", instructions="...")
        - result = Runner.run_sync(agent, "…")
        
        Detection keys:
        - Import module literally named agents
        - Agent(...) constructor call
        - Runner.run_sync(...) / Runner.run(...) with that Agent
        """
        agents = []
        
        if not self._has_import("agents"):
            return agents
        
        # Look for Agent constructor calls
        agent_calls = self._find_constructor_calls(["Agent"])
        
        # Look for Runner.run_sync calls
        runner_calls = self._find_method_calls(["run_sync", "run"])
        
        all_calls = agent_calls + runner_calls
        
        for call_site in all_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has agents imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "agents"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.OpenAIAgentsSDK,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"OpenAI Agents SDK agent via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_langroid_agent(self)->List[Agent]:
        """Detect Langroid agents.
        
        Agent construction pattern:
        - Langroid commonly uses a config + agent:
          import langroid as lr
          config = lr.ChatAgentConfig(...)
          agent = lr.ChatAgent(config)
        - Often followed by a Task wrapper:
          from langroid.agent.task import Task
          task = Task(agent, ...)
          task.run() / task.run_async()
        
        Detection keys:
        - Imports containing langroid
        - ChatAgentConfig(...) then ChatAgent(...)
        - Task(agent, ...)
        """
        agents = []
        
        if not self._has_import("langroid"):
            return agents
        
        # Look for ChatAgent constructor calls
        agent_calls = self._find_constructor_calls(["ChatAgent", "ChatAgentConfig"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has langroid imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "langroid"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.Langroid,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description=f"Langroid agent created via {call_site.method_name}",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_camel_agent(self)->List[Agent]:
        """Detect CAMEL agents.
        
        Agent creation pattern:
        - from camel.agents import ChatAgent
        - agent = ChatAgent(...)
        - Often includes message helpers:
          from camel.messages import BaseMessage as bm
          sys_msg = bm.make_assistant_message(...)
          agent = ChatAgent(system_message=sys_msg, ...)
        
        Detection keys:
        - Imports containing camel.agents
        - ChatAgent(...) constructor calls
        - bm.make_assistant_message(...) nearby is a strong hint
        """
        agents = []
        
        if not self._has_import("camel"):
            return agents
        
        # Look for ChatAgent constructor calls
        agent_calls = self._find_constructor_calls(["ChatAgent"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has camel imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "camel"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.CAMEL,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="CAMEL agent created via ChatAgent",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_letta_agent(self)->List[Agent]:
        """Detect Letta agents.
        
        Letta is very distinctive: agents are typically created via the client.
        
        Creation pattern:
        - from letta_client import Letta
        - client = Letta(api_key=...)
        - agent_state = client.agents.create(model=..., embedding=..., memory_blocks=[...], tools=[...])
        
        Detection keys:
        - Imports containing letta_client
        - Call pattern: client.agents.create(...) (this is your "agent creation" signature)
        """
        agents = []
        
        if not self._has_import("letta"):
            return agents
        
        # Look for create method calls (client.agents.create)
        create_calls = self._find_method_calls(["create"])
        
        for call_site in create_calls:
            # Check if this is likely a Letta agent creation (has "agents" in context)
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has letta imports
                module = self._get_module_for_function(func)
                if not module or not self._module_has_import(module, "letta"):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                # Simple heuristic: if we see "create" and have letta imports, it's likely an agent
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.Letta,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=[],
                    description="Letta agent created via client.agents.create",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents

    def detect_claude_agent_sdk_agent(self)->List[Agent]:
        """Detect Claude Agent SDK agents.
        
        Claude Agent SDK patterns:
        - from anthropic.agents import Agent
        - from claude_agent_sdk import Agent
        - agent = Agent(name="...", model="claude-3-5-sonnet-20241022", tools=[...])
        - agent.run(input="...")
        
        Detection keys:
        - Imports containing anthropic.agents or claude_agent_sdk
        - Agent(...) constructor calls with Claude-specific parameters
        - run() method calls on agent instances
        """
        agents = []
        
        if not (self._has_import("anthropic.agents") or self._has_import("claude_agent_sdk")):
            return agents
        
        # Look for Agent constructor calls
        agent_calls = self._find_constructor_calls(["Agent"])
        
        for call_site in agent_calls:
            func = self._get_function_for_callsite(call_site)
            if func:
                # Verify the module has Claude Agent SDK imports
                module = self._get_module_for_function(func)
                if not module or not (self._module_has_import(module, "anthropic.agents") or 
                                     self._module_has_import(module, "claude_agent_sdk")):
                    continue
                
                # Check if agent already exists
                if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
                    continue
                
                # Collect call details
                constructor_calls, application_calls, library_calls = self._collect_call_details(func)
                
                # Detect tools used in this agent
                agent_tools = self._detect_agent_tools(func)
                
                # Calculate metrics
                ncloc = self._calculate_ncloc(func)
                complexity = self._calculate_cyclomatic_complexity(func)
                num_objects = len(constructor_calls)
                    
                agent = Agent(
                    name=func.name,
                    agent_type=AgentType.FACTORY,
                    framework=AgenticFramework.ClaudeAgentSDK,
                    qualified_class_name="",
                    qualified_module_name=func.qualified_module_name,
                    method_signature=func.qualified_name,
                    tools=agent_tools,
                    description="Claude Agent SDK agent created via Agent constructor",
                    ncloc=ncloc,
                    ncloc_with_helpers=ncloc,
                    cyclomatic_complexity=complexity,
                    cyclomatic_complexity_with_helpers=complexity,
                    number_of_objects_created=num_objects,
                    constructor_call_details=constructor_calls,
                    application_call_details=application_calls,
                    library_call_details=library_calls,
                )
                agents.append(agent)
        
        return agents
        
    def detect_fastmcp_agent(self)->List[Agent]:
        """Detect FastMCP agents.
        
        FastMCP patterns:
        - from fastmcp import FastMCP
        - from mcp import Server
        - server = FastMCP(name="...")
        - @server.tool() decorator for tools
        
        Detection keys:
        - Imports containing fastmcp or mcp
        - FastMCP(...) or Server(...) constructor calls
        - @server.tool() decorated functions
        """
        agents = []
        
        # if not (self._has_import("fastmcp") or self._has_import("mcp")):
        #     return agents
        #
        # # Look for FastMCP/Server constructor calls
        # agent_calls = self._find_constructor_calls(["FastMCP", "Server"])
        #
        # for call_site in agent_calls:
        #     func = self._get_function_for_callsite(call_site)
        #     if func:
        #         # Verify the module has FastMCP imports
        #         module = self._get_module_for_function(func)
        #         if not module or not (self._module_has_import(module, "fastmcp") or
        #                              self._module_has_import(module, "mcp")):
        #             continue
        #
        #         # Check if agent already exists
        #         if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
        #             continue
        #
        #         # Collect call details
        #         constructor_calls, application_calls, library_calls = self._collect_call_details(func)
        #
        #         # Detect tools used in this agent
        #         agent_tools = self._detect_agent_tools(func)
        #
        #         # Calculate metrics
        #         ncloc = self._calculate_ncloc(func)
        #         complexity = self._calculate_cyclomatic_complexity(func)
        #         num_objects = len(constructor_calls)
        #
        #         agent = Agent(
        #             name=func.name,
        #             agent_type=AgentType.FACTORY,
        #             framework=AgenticFramework.FastMCP,
        #             qualified_class_name="",
        #             qualified_module_name=func.qualified_module_name,
        #             method_signature=func.qualified_name,
        #             tools=agent_tools,
        #             description="FastMCP server/agent created via FastMCP or Server constructor",
        #             ncloc=ncloc,
        #             ncloc_with_helpers=ncloc,
        #             cyclomatic_complexity=complexity,
        #             cyclomatic_complexity_with_helpers=complexity,
        #             number_of_objects_created=num_objects,
        #             constructor_call_details=constructor_calls,
        #             application_call_details=application_calls,
        #             library_call_details=library_calls,
        #         )
        #         agents.append(agent)
        #
        return agents
    
    def detect_mcp_agent(self)->List[Agent]:
        """Detect MCP (Model Context Protocol) agents using Python's builtin MCP library.
        
        MCP patterns (builtin library):
        - from mcp.server import Server
        - from mcp.server.stdio import stdio_server
        - from mcp.server.sse import sse_server
        - server = Server(name="...")
        - @server.tool() decorator for tools
        - @server.list_tools() for tool listing
        - @server.call_tool() for tool execution
        
        Detection keys:
        - Imports containing mcp.server, mcp.types, mcp.client
        - Server(...) constructor calls
        - stdio_server() or sse_server() calls
        - @server.tool() decorated functions
        """
        agents = []
        
        # Check for MCP builtin library imports (not FastMCP)
        # if not (self._has_import("mcp.server") or
        #         self._has_import("mcp.types") or
        #         self._has_import("mcp.client")):
        #     return agents
        #
        # # Look for Server constructor calls
        # agent_calls = self._find_constructor_calls(["Server"])
        #
        # for call_site in agent_calls:
        #     func = self._get_function_for_callsite(call_site)
        #     if func:
        #         # Verify the module has MCP builtin library imports (not FastMCP)
        #         module = self._get_module_for_function(func)
        #         if not module:
        #             continue
        #
        #         # Must have mcp.server imports, but NOT fastmcp imports
        #         has_mcp_server = (self._module_has_import(module, "mcp.server") or
        #                          self._module_has_import(module, "mcp.types") or
        #                          self._module_has_import(module, "mcp.client"))
        #         has_fastmcp = self._module_has_import(module, "fastmcp")
        #
        #         if not has_mcp_server or has_fastmcp:
        #             continue
        #
        #         # Check if agent already exists
        #         if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
        #             continue
        #
        #         # Collect call details
        #         constructor_calls, application_calls, library_calls = self._collect_call_details(func)
        #
        #         # Detect tools used in this agent
        #         agent_tools = self._detect_agent_tools(func)
        #
        #         # Calculate metrics
        #         ncloc = self._calculate_ncloc(func)
        #         complexity = self._calculate_cyclomatic_complexity(func)
        #         num_objects = len(constructor_calls)
        #
        #         agent = Agent(
        #             name=func.name,
        #             agent_type=AgentType.FACTORY,
        #             framework=AgenticFramework.MCP,
        #             qualified_class_name="",
        #             qualified_module_name=func.qualified_module_name,
        #             method_signature=func.qualified_name,
        #             tools=agent_tools,
        #             description="MCP server/agent created via builtin MCP Server constructor",
        #             ncloc=ncloc,
        #             ncloc_with_helpers=ncloc,
        #             cyclomatic_complexity=complexity,
        #             cyclomatic_complexity_with_helpers=complexity,
        #             number_of_objects_created=num_objects,
        #             constructor_call_details=constructor_calls,
        #             application_call_details=application_calls,
        #             library_call_details=library_calls,
        #         )
        #         agents.append(agent)
        #
        # # Also look for stdio_server() and sse_server() function calls
        # server_calls = self._find_method_calls(["stdio_server", "sse_server"])
        #
        # for call_site in server_calls:
        #     func = self._get_function_for_callsite(call_site)
        #     if func:
        #         module = self._get_module_for_function(func)
        #         if not module:
        #             continue
        #
        #         # Verify MCP imports
        #         has_mcp_server = (self._module_has_import(module, "mcp.server.stdio") or
        #                          self._module_has_import(module, "mcp.server.sse"))
        #         has_fastmcp = self._module_has_import(module, "fastmcp")
        #
        #         if not has_mcp_server or has_fastmcp:
        #             continue
        #
        #         # Check if agent already exists
        #         if self._agent_exists(agents, func.qualified_name, func.qualified_module_name):
        #             continue
        #
        #         # Collect call details
        #         constructor_calls, application_calls, library_calls = self._collect_call_details(func)
        #
        #         # Detect tools
        #         agent_tools = self._detect_agent_tools(func)
        #
        #         # Calculate metrics
        #         ncloc = self._calculate_ncloc(func)
        #         complexity = self._calculate_cyclomatic_complexity(func)
        #         num_objects = len(constructor_calls)
        #
        #         agent = Agent(
        #             name=func.name,
        #             agent_type=AgentType.FACTORY,
        #             framework=AgenticFramework.MCP,
        #             qualified_class_name="",
        #             qualified_module_name=func.qualified_module_name,
        #             method_signature=func.qualified_name,
        #             tools=agent_tools,
        #             description=f"MCP server/agent created via {call_site.method_name}()",
        #             ncloc=ncloc,
        #             ncloc_with_helpers=ncloc,
        #             cyclomatic_complexity=complexity,
        #             cyclomatic_complexity_with_helpers=complexity,
        #             number_of_objects_created=num_objects,
        #             constructor_call_details=constructor_calls,
        #             application_call_details=application_calls,
        #             library_call_details=library_calls,
        #         )
        #         agents.append(agent)
        #
        return agents

