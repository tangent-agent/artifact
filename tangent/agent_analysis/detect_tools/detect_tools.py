import ast
from pathlib import Path
from typing import List, Set

from tangent.code_analysis.code_analysis import PythonAnalysis
from tangent.code_analysis.model.model import PyCallSite, PyFunction, PyModule, PyClass
from tangent.agent_analysis.model.models import Tool
from tangent.utils.constants import AgenticFramework, ToolType


class DetectTools:
    """Detect tools used in agentic applications.
    
    Tools can be defined in various ways:
    1. Decorator-based tools: @tool, @langchain_tool, etc.
    2. Class-based tools: Classes inheriting from BaseTool, Tool, etc.
    3. Function-based tools: Functions passed to tool lists
    4. Lambda tools: Lambda functions used as tools
    """
    
    # Framework-specific import patterns for tool validation
    FRAMEWORK_TOOL_IMPORTS = {
        AgenticFramework.LangChain: [
            "langchain", "langchain_core", "langchain_community"
        ],
        AgenticFramework.LangGraph: ["langgraph"],
        AgenticFramework.CrewAI: ["crewai"],
        AgenticFramework.AutoGen: ["autogen", "autogen_core", "autogen_agentchat"],
        AgenticFramework.LlamaIndex: ["llama_index"],
        AgenticFramework.OpenAISwarm: ["swarm"],
        AgenticFramework.OpenAIAgentsSDK: ["agents"],
        AgenticFramework.Langroid: ["langroid"],
        AgenticFramework.CAMEL: ["camel"],
        AgenticFramework.Letta: ["letta", "letta_client"],
        AgenticFramework.MCP: ["mcp", "mcp.server", "mcp.types", "mcp.client"],
        AgenticFramework.FastMCP: ["fastmcp"],
        AgenticFramework.ClaudeAgentSDK: ["claude_agent_sdk", "anthropic.agents"],
    }
    
    def __init__(self, analysis: PythonAnalysis):
        self.analysis = analysis
        self._detected_tools: Set[str] = set()  # Track unique tools by qualified name
    
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
    
    def _module_has_framework_imports(self, module: PyModule, frameworks: List[AgenticFramework]) -> bool:
        """Check if a module has imports from any of the detected frameworks.
        
        This ensures that tools are only identified in modules that actually use agentic frameworks.
        """
        if not module or not module.imports:
            return False
        
        for framework in frameworks:
            if framework in self.FRAMEWORK_TOOL_IMPORTS:
                for import_pattern in self.FRAMEWORK_TOOL_IMPORTS[framework]:
                    if self._module_has_import(module, import_pattern):
                        return True
        return False
    
    def _get_module_for_function(self, func: PyFunction) -> PyModule | None:
        """Get the module containing a function."""
        for module in self.analysis.get_modules():
            if func in module.functions:
                return module
            for cls in module.classes:
                if func in cls.methods:
                    return module
        return None
    
    def _get_module_for_class(self, cls: PyClass) -> PyModule | None:
        """Get the module containing a class."""
        for module in self.analysis.get_modules():
            if cls in module.classes:
                return module
        return None
    
    def _extract_docstring(self, func: PyFunction) -> str:
        """Extract docstring from a function."""
        if func.docstring:
            return func.docstring.strip()
        return ""
    
    def _extract_class_docstring(self, cls: PyClass) -> str:
        """Extract docstring from a class."""
        if cls.docstring:
            return cls.docstring.strip()
        return ""
    
    def _is_tool_decorator(self, decorator_expr: str) -> bool:
        """Check if a decorator expression indicates a tool (excluding MCP tools)."""
        # Exclude MCP tools
        if "mcp.tool" in decorator_expr or "mcp_server.tool" in decorator_expr:
            return False
        
        tool_decorators = [
            "tool", "langchain_tool", "structured_tool",
            "Tool", "StructuredTool", "BaseTool",
            "function_tool", "FunctionTool"
        ]
        return any(td in decorator_expr for td in tool_decorators)
    
    # Decorator expressions that hint at a specific framework
    DECORATOR_FRAMEWORK_HINTS = {
        AgenticFramework.LangChain: ["langchain_tool", "structured_tool", "StructuredTool"],
        AgenticFramework.CrewAI: ["crewai"],
        AgenticFramework.LlamaIndex: ["llama_index", "FunctionTool"],
        AgenticFramework.OpenAIAgentsSDK: ["function_tool"],
        AgenticFramework.AutoGen: ["autogen"],
        AgenticFramework.Langroid: ["langroid"],
        AgenticFramework.CAMEL: ["camel"],
        AgenticFramework.Letta: ["letta"],
        AgenticFramework.FastMCP: ["fastmcp"],
    }

    # Base class names mapped to their corresponding framework
    BASE_CLASS_FRAMEWORK_HINTS = {
        AgenticFramework.LangChain: ["BaseTool", "StructuredTool", "LangChainTool"],
        AgenticFramework.LangGraph: ["BaseTool", "StructuredTool"],
        AgenticFramework.CrewAI: ["BaseTool"],
        AgenticFramework.LlamaIndex: ["FunctionTool", "BaseTool"],
        AgenticFramework.OpenAIAgentsSDK: ["FunctionTool", "Tool"],
        AgenticFramework.AutoGen: ["Tool"],
        AgenticFramework.Langroid: ["Tool"],
        AgenticFramework.CAMEL: ["Tool"],
        AgenticFramework.Letta: ["Tool"],
    }

    def _is_tool_base_class(self, base_name: str) -> bool:
        """Check if a base class indicates a tool (any framework)."""
        return any(
            any(tb in base_name for tb in bases)
            for bases in self.BASE_CLASS_FRAMEWORK_HINTS.values()
        )

    def _infer_framework_from_base_class(
        self,
        base_name: str,
        module: "PyModule",
        frameworks: List[AgenticFramework] | None = None,
    ) -> AgenticFramework:
        """Infer framework from a base class name, cross-checked with module imports.

        Priority:
        1. Base class name matches a framework hint AND module has that framework's imports
        2. Base class name matches a framework hint (import check skipped)
        3. Fall back to import-only inference
        """
        candidate_frameworks = frameworks if frameworks is not None else list(self.BASE_CLASS_FRAMEWORK_HINTS.keys())

        # First pass: require both base-class hint AND matching import
        for fw in candidate_frameworks:
            if fw in self.BASE_CLASS_FRAMEWORK_HINTS:
                if any(tb in base_name for tb in self.BASE_CLASS_FRAMEWORK_HINTS[fw]):
                    if fw in self.FRAMEWORK_TOOL_IMPORTS:
                        for import_pattern in self.FRAMEWORK_TOOL_IMPORTS[fw]:
                            if self._module_has_import(module, import_pattern):
                                return fw

        # Second pass: base-class hint alone (import not found but name is specific enough)
        for fw in candidate_frameworks:
            if fw in self.BASE_CLASS_FRAMEWORK_HINTS:
                if any(tb in base_name for tb in self.BASE_CLASS_FRAMEWORK_HINTS[fw]):
                    return fw

        return AgenticFramework.Unknown

    def _infer_framework_from_module(
        self,
        module: "PyModule",
        frameworks: List[AgenticFramework] | None = None,
        decorator_expr: str | None = None,
    ) -> AgenticFramework:
        """Infer the agentic framework for a tool based on module imports and/or decorator.

        Priority:
        1. Decorator expression hints (most specific)
        2. Module-level import matching against detected frameworks
        3. Fall back to AgenticFramework.Unknown
        """
        # 1. Check decorator expression for framework-specific hints
        if decorator_expr:
            for fw, hints in self.DECORATOR_FRAMEWORK_HINTS.items():
                if any(hint in decorator_expr for hint in hints):
                    return fw

        # 2. Match module imports against the detected (or all) frameworks
        candidate_frameworks = frameworks if frameworks is not None else list(self.FRAMEWORK_TOOL_IMPORTS.keys())
        for fw in candidate_frameworks:
            if fw in self.FRAMEWORK_TOOL_IMPORTS:
                for import_pattern in self.FRAMEWORK_TOOL_IMPORTS[fw]:
                    if self._module_has_import(module, import_pattern):
                        return fw

        return AgenticFramework.Unknown
    
    def detect_decorator_tools(self, frameworks: List[AgenticFramework] | None = None) -> List[Tool]:
        """Detect tools defined using decorators.
        
        Args:
            frameworks: List of detected agentic frameworks to validate against.
                       If None, skips framework validation (for backward compatibility).
        """
        tools = []
        
        for func in self.analysis.iter_functions():
            # Check if function has tool decorators
            has_tool_decorator = any(
                self._is_tool_decorator(dec.expression)
                for dec in func.decorators
            )
            
            if has_tool_decorator:
                module = self._get_module_for_function(func)
                if not module:
                    continue
                
                # VALIDATION: Ensure module has framework-related imports (if frameworks provided)
                if frameworks is not None and not self._module_has_framework_imports(module, frameworks):
                    continue
                
                # Create unique identifier
                tool_id = f"{func.qualified_module_name}.{func.name}"
                if tool_id in self._detected_tools:
                    continue
                self._detected_tools.add(tool_id)
                
                # Extract description from docstring
                description = self._extract_docstring(func)
                
                # Infer framework from decorator expression and module imports
                decorator_expr = next(
                    (dec.expression for dec in func.decorators if self._is_tool_decorator(dec.expression)),
                    None,
                )
                framework = self._infer_framework_from_module(module, frameworks, decorator_expr)
                
                tool = Tool(
                    framework=framework,
                    tool_name=func.name,
                    tool_type=ToolType.DECORATOR_TOOL,
                    tool_description=description if description else None,
                    tool_binding_method=func.qualified_name,
                    tool_binding_module=func.qualified_module_name
                )
                tools.append(tool)
        
        return tools
    
    def detect_class_tools(self, frameworks: List[AgenticFramework] | None = None) -> List[Tool]:
        """Detect tools defined as classes (inheriting from tool base classes).
        
        Args:
            frameworks: List of detected agentic frameworks to validate against.
                       If None, skips framework validation (for backward compatibility).
        """
        tools = []
        
        for module in self.analysis.get_modules():
            # VALIDATION: Skip modules without framework imports (if frameworks provided)
            if frameworks is not None and not self._module_has_framework_imports(module, frameworks):
                continue
            
            for cls in module.classes:
                # Check if class inherits from tool base classes
                has_tool_base = any(
                    self._is_tool_base_class(base)
                    for base in cls.bases
                )
                
                if has_tool_base:
                    # Create unique identifier
                    tool_id = f"{cls.qualified_name}"
                    if tool_id in self._detected_tools:
                        continue
                    self._detected_tools.add(tool_id)
                    
                    # Extract description from class docstring
                    description = self._extract_class_docstring(cls)
                    
                    # Infer framework from the base class name and module imports
                    base_hint = next(
                        (base for base in cls.bases if self._is_tool_base_class(base)),
                        None,
                    )
                    if base_hint:
                        framework = self._infer_framework_from_base_class(base_hint, module, frameworks)
                    else:
                        framework = self._infer_framework_from_module(module, frameworks)
                    
                    tool = Tool(
                        framework=framework,
                        tool_name=cls.class_name,
                        tool_type=ToolType.CLASS_TOOL,
                        tool_description=description if description else None,
                        tool_binding_method=cls.qualified_name,
                        tool_binding_module=module.qualified_name
                    )
                    tools.append(tool)
        
        return tools
    
    def detect_function_tools(self, frameworks: List[AgenticFramework] | None = None) -> List[Tool]:
        """Detect tools passed as functions to agent tool lists.
        
        This looks for patterns like:
        - tools=[function1, function2]
        - agent.bind_tools([func1, func2])
        
        Args:
            frameworks: List of detected agentic frameworks to validate against.
                       If None, skips framework validation (for backward compatibility).
        """
        tools = []
        
        # Look for common tool-related method calls
        tool_methods = ["bind_tools", "with_tools", "add_tool", "register_tool"]
        
        for call_site in self.analysis.get_call_sites():
            if call_site.method_name in tool_methods:
                # Try to extract function references from arguments
                for arg_expr in call_site.argument_expr:
                    # Simple heuristic: if argument looks like a function name
                    # (not a string, not a constructor call)
                    if arg_expr and not arg_expr.startswith('"') and not arg_expr.startswith("'"):
                        # Look for the function in our analysis
                        for func in self.analysis.iter_functions():
                            if func.name in arg_expr:
                                module = self._get_module_for_function(func)
                                if not module:
                                    continue
                                
                                # VALIDATION: Ensure module has framework-related imports (if frameworks provided)
                                if frameworks is not None and not self._module_has_framework_imports(module, frameworks):
                                    continue
                                
                                tool_id = f"{func.qualified_module_name}.{func.name}"
                                if tool_id in self._detected_tools:
                                    continue
                                self._detected_tools.add(tool_id)
                                
                                description = self._extract_docstring(func)
                                
                                framework = self._infer_framework_from_module(module, frameworks)
                                
                                tool = Tool(
                                    framework=framework,
                                    tool_name=func.name,
                                    tool_type=ToolType.FUNCTION_TOOL,
                                    tool_description=description if description else None,
                                    tool_binding_method=func.qualified_name,
                                    tool_binding_module=func.qualified_module_name
                                )
                                tools.append(tool)
                                break
        
        return tools
    
    def detect_lambda_tools(self) -> List[Tool]:
        """Detect lambda functions used as tools.
        
        This is more challenging and requires AST analysis.
        """
        tools = []
        
        # Look for lambda expressions in tool-related contexts
        for module in self.analysis.get_modules():
            try:
                source = Path(module.file_path).read_text(encoding='utf-8')
                tree = ast.parse(source)
                
                # Walk the AST looking for lambda in tool contexts
                for node in ast.walk(tree):
                    # Look for assignments or calls with lambda
                    if isinstance(node, ast.Lambda):
                        # Try to determine if this lambda is used as a tool
                        # This is a heuristic - we look for lambda in list contexts
                        # or as arguments to tool-related functions
                        parent = self._find_parent(tree, node)
                        if parent and self._is_tool_context(parent):
                            # Create a synthetic tool name
                            tool_name = f"lambda_tool_{node.lineno}"
                            tool_id = f"{module.qualified_name}.{tool_name}"
                            
                            if tool_id in self._detected_tools:
                                continue
                            self._detected_tools.add(tool_id)
                            
                            framework = self._infer_framework_from_module(module)
                            
                            tool = Tool(
                                framework=framework,
                                tool_name=tool_name,
                                tool_type=ToolType.LAMBDA_TOOL,
                                tool_description=f"Lambda tool at line {node.lineno}",
                                tool_binding_method=tool_name,
                                tool_binding_module=module.qualified_name
                            )
                            tools.append(tool)
            except Exception:
                # Skip files that can't be parsed
                continue
        
        return tools
    
    def _find_parent(self, tree: ast.AST, target: ast.AST) -> ast.AST | None:
        """Find the parent node of a target node in the AST."""
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                if child is target:
                    return node
        return None
    
    def _is_tool_context(self, node: ast.AST) -> bool:
        """Check if a node represents a tool-related context."""
        # Check if node is a call to tool-related functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                tool_functions = ["bind_tools", "with_tools", "add_tool", "register_tool"]
                return node.func.id in tool_functions
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr in ["bind_tools", "with_tools", "add_tool", "register_tool"]
        
        # Check if node is a list that might contain tools
        if isinstance(node, ast.List):
            return True
        
        return False
    
    def detect_mcp_tools(self) -> List[Tool]:
        """Detect MCP (Model Context Protocol) tools using Python's builtin MCP library.
        
        MCP tools (builtin library) are typically defined using:
        - @server.tool() decorator
        - @server.list_tools() handler
        - @server.call_tool() handler
        - Functions decorated with @tool from mcp.server
        """
        tools = []
        
        # Check for MCP builtin library imports (not FastMCP)
        if not (self._has_import("mcp.server") or
                self._has_import("mcp.types") or
                self._has_import("mcp.client")):
            return tools
        
        # Detect @server.tool() or @tool decorated functions
        for func in self.analysis.iter_functions():
            has_mcp_decorator = any(
                "server.tool" in dec.expression or
                "@tool" in dec.expression or
                "mcp.tool" in dec.expression
                for dec in func.decorators
            )
            
            if has_mcp_decorator:
                module = self._get_module_for_function(func)
                if not module:
                    continue
                
                # Verify module has MCP builtin library imports (not FastMCP)
                has_mcp_server = (self._module_has_import(module, "mcp.server") or
                                 self._module_has_import(module, "mcp.types"))
                has_fastmcp = self._module_has_import(module, "fastmcp")
                
                if not has_mcp_server or has_fastmcp:
                    continue
                
                tool_id = f"{func.qualified_module_name}.{func.name}"
                if tool_id in self._detected_tools:
                    continue
                self._detected_tools.add(tool_id)
                
                description = self._extract_docstring(func)
                
                tool = Tool(
                    framework=AgenticFramework.MCP,
                    tool_name=func.name,
                    tool_type=ToolType.MCP_TOOL,
                    tool_description=description if description else None,
                    tool_binding_method=func.qualified_name,
                    tool_binding_module=func.qualified_module_name
                )
                tools.append(tool)
        
        return tools
    
    def detect_claude_agent_sdk_tools(self) -> List[Tool]:
        """Detect Claude Agent SDK tools.
        
        Claude Agent SDK tools are typically defined as:
        - Functions decorated with @tool
        - Classes with tool-like structure
        - Functions passed to Agent(tools=[...])
        
        Note: Claude Agent SDK tools already have built-in framework validation via import checks.
        """
        tools = []
        
        # VALIDATION: Check for Claude Agent SDK framework imports at global level
        if not (self._has_import("anthropic.agents") or self._has_import("claude_agent_sdk")):
            return tools
        
        # Detect decorator-based tools (similar to other frameworks)
        for func in self.analysis.iter_functions():
            has_tool_decorator = any(
                "tool" in dec.expression.lower()
                for dec in func.decorators
            )
            
            if has_tool_decorator:
                module = self._get_module_for_function(func)
                if not module:
                    continue
                
                # VALIDATION: Verify module has Claude Agent SDK framework imports
                if not (self._module_has_import(module, "anthropic.agents") or
                       self._module_has_import(module, "claude_agent_sdk")):
                    continue
                
                tool_id = f"{func.qualified_module_name}.{func.name}"
                if tool_id in self._detected_tools:
                    continue
                self._detected_tools.add(tool_id)
                
                description = self._extract_docstring(func)
                
                tool = Tool(
                    framework=AgenticFramework.ClaudeAgentSDK,
                    tool_name=func.name,
                    tool_type=ToolType.DECORATOR_TOOL,
                    tool_description=description if description else None,
                    tool_binding_method=func.qualified_name,
                    tool_binding_module=func.qualified_module_name
                )
                tools.append(tool)
        
        return tools
    
    def detect_tools(self, frameworks: List[AgenticFramework]) -> List[Tool]:
        """Detect all tools in the codebase.
        
        Args:
            frameworks: List of detected agentic frameworks
            
        Returns:
            List of detected tools
        """
        all_tools = []
        
        # Only detect tools if we have relevant frameworks
        relevant_frameworks = [
            AgenticFramework.LangChain,
            AgenticFramework.LangGraph,
            AgenticFramework.CrewAI,
            AgenticFramework.AutoGen,
            AgenticFramework.LlamaIndex,
            AgenticFramework.OpenAISwarm,
            AgenticFramework.OpenAIAgentsSDK,
            AgenticFramework.Langroid,
            AgenticFramework.CAMEL,
            AgenticFramework.Letta,
            AgenticFramework.MCP,
            AgenticFramework.FastMCP,
            AgenticFramework.ClaudeAgentSDK
        ]
        
        if not any(fw in frameworks for fw in relevant_frameworks):
            return all_tools
        
        # Detect different types of tools with framework validation
        all_tools.extend(self.detect_decorator_tools(frameworks))
        all_tools.extend(self.detect_class_tools(frameworks))
        all_tools.extend(self.detect_function_tools(frameworks))
        all_tools.extend(self.detect_lambda_tools())
        
        # Detect MCP tools if MCP builtin library framework is present
        if AgenticFramework.MCP in frameworks:
            all_tools.extend(self.detect_mcp_tools())
        
        # Detect FastMCP tools if FastMCP framework is present
        # Note: FastMCP tools are detected via decorator_tools since they use standard decorators
        
        # Detect Claude Agent SDK tools if framework is present
        if AgenticFramework.ClaudeAgentSDK in frameworks:
            all_tools.extend(self.detect_claude_agent_sdk_tools())
        
        return all_tools


