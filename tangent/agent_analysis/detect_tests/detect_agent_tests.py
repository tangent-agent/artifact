from typing import List, Set



from tangent.agent_analysis.model.models import Agent, AgentTest, TestMethod, Fixture
from tangent.code_analysis.code_analysis import PythonAnalysis
from tangent.code_analysis.model.model import PyModule, PyFunction, PyClass
from tangent.utils.constants import AgenticFramework, TestingFramework


class DetectAgentTests:
    def __init__(self, analysis: PythonAnalysis,
                 agents: List[Agent], frameworks: List[AgenticFramework]):
        self.analysis = analysis
        self.agents = agents
        self.frameworks = frameworks

    def _is_test_module(self, module: PyModule) -> bool:
        """Check if a module is a test module."""
        return module.is_test or 'test' in module.file_path.lower()

    def _is_test_function(self, func: PyFunction) -> bool:
        """Check if a function is a test function."""
        return func.is_test or func.name.startswith('test_')

    def _module_imports_agent(self, module: PyModule, agent: Agent) -> bool:
        """Check if a module imports the agent's module or specific agent function."""
        if not module.imports:
            return False
        
        agent_module = agent.qualified_module_name
        # Remove .py extension if present for comparison with import statements
        agent_module_no_ext = agent_module.replace('.py', '') if agent_module.endswith('.py') else agent_module
        agent_name = agent.name
        
        for imp in module.imports:
            # Check if the agent's module is imported (with or without .py extension)
            if agent_module in imp.from_statement or agent_module_no_ext in imp.from_statement:
                # Check if the specific agent function is imported
                if agent_name in imp.imports or '*' in imp.imports:
                    return True
            # Check if the full module path is imported (with or without .py extension)
            if agent_module in imp.imports or agent_module_no_ext in imp.imports:
                return True
        
        return False

    def _function_calls_agent(self, func: PyFunction, agent: Agent) -> bool:
        """Check if a function calls the agent method."""
        agent_name = agent.name
        
        for call_site in func.call_sites:
            # Check if the call site method name matches the agent name
            if call_site.method_name == agent_name:
                return True
            # Check if the qualified name matches
            if agent.method_signature in call_site.method_signature:
                return True
        
        return False

    def _get_agents_used_in_function(self, func: PyFunction) -> List[Agent]:
        """Get list of agents that are called in this function."""
        agents_used = []
        for agent in self.agents:
            if self._function_calls_agent(func, agent):
                agents_used.append(agent)
        return agents_used

    def detect_agent_tests(self) -> List[AgentTest]:
        """
        Detect test modules and functions that import and call agent methods.
        
        Returns a list of AgentTest objects containing:
        - Test class/module information
        - Test methods that use agents
        - Fixtures (setup/teardown methods)
        """
        agent_tests: List[AgentTest] = []
        
        # Iterate through all modules
        for module in self.analysis.get_modules():
            # Skip non-test modules
            if not self._is_test_module(module):
                continue
            
            # Check if this module imports any of our detected agents
            imports_agents = any(self._module_imports_agent(module, agent) for agent in self.agents)
            if not imports_agents:
                continue
            
            # Process test classes in the module
            for cls in module.classes:
                if not cls.is_test_class:
                    continue
                
                test_methods: List[TestMethod] = []
                fixtures: List[Fixture] = []
                
                # Process methods in the test class
                for method in cls.methods:
                    # Check if this is a test method
                    if self._is_test_function(method):
                        # Get agents used in this test method
                        agents_used = self._get_agents_used_in_function(method)
                        
                        if agents_used:
                            test_method = TestMethod(
                                agents=agents_used,
                                method_signature=method.qualified_name,
                                method_declaration=method.name,
                                annotations=[d.expression for d in method.decorators],
                                ncloc=method.end_line - method.start_line + 1 if method.start_line > 0 else 0,
                            )
                            test_methods.append(test_method)
                    
                    # Check for fixture methods (setup/teardown)
                    elif any(decorator.expression in ['setUp', 'tearDown', 'setup', 'teardown', 'pytest.fixture']
                            for decorator in method.decorators):
                        fixture = Fixture(
                            is_setup='setup' in method.name.lower() or 'setUp' in method.name,
                            is_teardown='teardown' in method.name.lower() or 'tearDown' in method.name,
                            qualified_class_name=cls.qualified_name,
                            qualified_module_name=module.qualified_name,
                            method_signature=method.qualified_name,
                            method_body="",  # We don't extract method body
                            ncloc=method.end_line - method.start_line + 1 if method.start_line > 0 else 0,
                        )
                        fixtures.append(fixture)
                
                # Create AgentTest if we found test methods
                if test_methods:
                    agent_test = AgentTest(
                        qualified_class_name=cls.qualified_name,
                        qualified_module_name=module.qualified_name,
                        fixtures=fixtures,
                        test_methods=test_methods,
                        testing_frameworks=[TestingFramework.PyUnit],  # Default to PyUnit
                    )
                    agent_tests.append(agent_test)
            
            # Process standalone test functions (not in a class)
            standalone_test_methods: List[TestMethod] = []
            for func in module.functions:
                if self._is_test_function(func):
                    agents_used = self._get_agents_used_in_function(func)
                    
                    if agents_used:
                        test_method = TestMethod(
                            agents=agents_used,
                            method_signature=func.qualified_name,
                            method_declaration=func.name,
                            annotations=[d.expression for d in func.decorators],
                            ncloc=func.end_line - func.start_line + 1 if func.start_line > 0 else 0,
                        )
                        standalone_test_methods.append(test_method)
            
            # Create AgentTest for standalone test functions
            if standalone_test_methods:
                agent_test = AgentTest(
                    qualified_class_name="",  # No class for standalone functions
                    qualified_module_name=module.qualified_name,
                    fixtures=[],
                    test_methods=standalone_test_methods,
                    testing_frameworks=[TestingFramework.PyUnit],
                )
                agent_tests.append(agent_test)
        
        return agent_tests