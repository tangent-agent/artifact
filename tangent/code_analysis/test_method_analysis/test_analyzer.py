"""Test method analysis module for extracting test complexity metrics.

This module provides functionality to analyze test methods and populate the
TestMethod model with complexity metrics, fixtures, helpers, and assertions.
"""

import ast
import re
from pathlib import Path
from typing import List, Optional, Set, Dict

from tangent.code_analysis.model.model import PyApplication, PyFunction, PyCallSite, PyClass
from tangent.code_analysis.model.test_complexity_model import (
    TestMethod,
    FixtureMethod,
    HelperMethod,
    Assertion,
)
from tangent.utils.constants import AssertionType

# Try to import tree-sitter based assert detector
try:
    from tangent.code_analysis.test_method_analysis.assert_detector import detect_assert_statements
    ASSERT_DETECTOR_AVAILABLE = True
except ImportError:
    ASSERT_DETECTOR_AVAILABLE = False
    detect_assert_statements = None


class TestMethodAnalyzer:
    """Analyzer for extracting test method complexity metrics."""

    def __init__(self, py_application: PyApplication, project_dir: Optional[str] = None):
        """Initialize the test method analyzer.
        
        Args:
            py_application: PyApplication model containing code analysis data
            project_dir: Optional project directory path for reading source files
        """
        self.py_application = py_application
        self.project_dir = Path(project_dir) if project_dir else None
        
        # Build lookup tables for efficient access
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build lookup tables for functions and classes."""
        self.function_lookup: Dict[str, PyFunction] = {}
        self.class_lookup: Dict[str, PyClass] = {}
        
        for module in self.py_application.symbol_table.values():
            # Index top-level functions
            for func in module.functions:
                self.function_lookup[func.qualified_name] = func
            
            # Index classes and their methods
            for cls in module.classes:
                self.class_lookup[cls.qualified_name] = cls
                for method in cls.methods:
                    self.function_lookup[method.qualified_name] = method

    def analyze_test_method(self, test_name: str) -> Optional[TestMethod]:
        """Analyze a test method and populate the TestMethod model.
        
        Args:
            test_name: Qualified name of the test method
            
        Returns:
            TestMethod model populated with complexity metrics, or None if not found
        """
        # Find the test function
        test_func = self.function_lookup.get(test_name)
        if not test_func:
            return None
        
        # Get the module containing this test
        module = self._get_module_for_function(test_func)
        if not module:
            return None
        
        # Extract basic information
        test_method = TestMethod(
            test_name=test_func.name,
            qualified_module_name=test_func.qualified_module_name,
            file_path=module.file_path,
            start_line=test_func.start_line,
            end_line=test_func.end_line,
            ncloc=self._calculate_ncloc(test_func),
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(test_func),
            number_of_constructor_call=self._count_constructor_calls(test_func),
            number_of_library_call=self._count_library_calls(test_func),
            number_of_application_call=self._count_application_calls(test_func),
            fixtures=[],
            helpers=[],
            assertions=[],
            is_async=test_func.is_async,
        )
        
        # Analyze fixtures
        test_method.fixtures = self._analyze_fixtures(test_func)
        test_method.number_of_fixtures_used = len(test_method.fixtures)
        
        # Analyze helper methods
        test_method.helpers = self._analyze_helpers(test_func)
        test_method.number_of_helper_methods = len(test_method.helpers)
        
        # Analyze assertions
        test_method.assertions = self._analyze_assertions(test_func)
        test_method.number_of_assertions = len(test_method.assertions)
        
        # Analyze mocking
        test_method.number_of_mocking_used = self._count_mocking(test_func)
        
        return test_method

    def _get_module_for_function(self, func: PyFunction):
        """Get the module containing a function."""
        for module in self.py_application.symbol_table.values():
            if func.qualified_module_name == module.qualified_name:
                return module
        return None

    def _calculate_ncloc(self, func: PyFunction) -> int:
        """Calculate non-comment lines of code.
        
        This is a simplified calculation based on line range.
        For more accurate results, parse the actual source.
        """
        if func.start_line < 0 or func.end_line < 0:
            return 0
        
        # Try to read source file for accurate count
        if self.project_dir:
            module = self._get_module_for_function(func)
            if module:
                try:
                    file_path = self.project_dir / module.file_path
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Count non-empty, non-comment lines
                            ncloc = 0
                            for i in range(func.start_line - 1, min(func.end_line, len(lines))):
                                line = lines[i].strip()
                                if line and not line.startswith('#'):
                                    ncloc += 1
                            return ncloc
                except Exception:
                    pass
        
        # Fallback: simple line count
        return max(0, func.end_line - func.start_line + 1)

    def _calculate_cyclomatic_complexity(self, func: PyFunction) -> int:
        """Calculate cyclomatic complexity.
        
        Simplified calculation: count decision points (if, for, while, except, etc.)
        """
        if not self.project_dir:
            return 1  # Default complexity
        
        module = self._get_module_for_function(func)
        if not module:
            return 1
        
        try:
            file_path = self.project_dir / module.file_path
            if not file_path.exists():
                return 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Find the function node
            func_node = self._find_function_node(tree, func)
            if not func_node:
                return 1
            
            # Count decision points
            complexity = 1  # Base complexity
            for node in ast.walk(func_node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                   ast.With, ast.Assert, ast.BoolOp)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    # Count each 'and'/'or' as a decision point
                    complexity += len(node.values) - 1
            
            return complexity
        except Exception:
            return 1

    def _find_function_node(self, tree: ast.AST, func: PyFunction) -> Optional[ast.FunctionDef]:
        """Find the AST node for a function."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == func.name and node.lineno == func.start_line:
                    return node
        return None

    def _count_constructor_calls(self, func: PyFunction) -> int:
        """Count constructor calls in the function."""
        return sum(1 for cs in func.call_sites if cs.is_constructor_call)

    def _count_library_calls(self, func: PyFunction) -> int:
        """Count library calls in the function."""
        return sum(1 for cs in func.call_sites if cs.is_library_call)

    def _count_application_calls(self, func: PyFunction) -> int:
        """Count application calls in the function."""
        return sum(1 for cs in func.call_sites if cs.is_application_call)

    def _analyze_fixtures(self, test_func: PyFunction) -> List[FixtureMethod]:
        """Analyze fixtures used by the test method.
        
        Supports both pytest fixtures (in parameters) and unittest fixtures (setUp/tearDown methods).
        """
        fixtures = []
        
        # 1. Check for pytest fixtures in parameters
        for param in test_func.parameters:
            # Skip 'self' and 'cls'
            if param.name in ('self', 'cls'):
                continue
            
            # Look for pytest fixture function
            fixture_func = self._find_pytest_fixture(param.name)
            if fixture_func:
                fixture = self._create_fixture_method(fixture_func)
                fixtures.append(fixture)
        
        # 2. Check for unittest setUp/tearDown methods in the same class
        if test_func.kind == "method":
            unittest_fixtures = self._find_unittest_fixtures(test_func)
            fixtures.extend(unittest_fixtures)
        
        return fixtures

    def _find_pytest_fixture(self, fixture_name: str) -> Optional[PyFunction]:
        """Find a pytest fixture function by name."""
        for func in self.function_lookup.values():
            if func.name == fixture_name:
                # Check if it has pytest.fixture decorator
                for decorator in func.decorators:
                    if 'fixture' in decorator.expression:
                        return func
        return None
    
    def _find_unittest_fixtures(self, test_func: PyFunction) -> List[FixtureMethod]:
        """Find unittest setUp/tearDown methods in the test's class.
        
        Args:
            test_func: The test method
            
        Returns:
            List of FixtureMethod objects for setUp/tearDown
        """
        fixtures = []
        
        # Extract class name from qualified name
        parts = test_func.qualified_name.rsplit('.', 1)
        if len(parts) < 2:
            return fixtures
        
        class_qualified_name = parts[0]
        test_class = self.class_lookup.get(class_qualified_name)
        
        if not test_class:
            return fixtures
        
        # Look for setUp and tearDown methods
        for method in test_class.methods:
            method_name_lower = method.name.lower()
            
            # Check for various setUp/tearDown naming conventions
            is_setup = method_name_lower in ('setup', 'setup_method', 'setup_class', 'setupclass')
            is_teardown = method_name_lower in ('teardown', 'teardown_method', 'teardown_class', 'teardownclass')
            
            if is_setup or is_teardown:
                fixture = self._create_fixture_method(method, is_setup=is_setup, is_teardown=is_teardown)
                fixtures.append(fixture)
        
        return fixtures

    def _create_fixture_method(self, func: PyFunction, is_setup: bool = False, is_teardown: bool = False) -> FixtureMethod:
        """Create a FixtureMethod from a PyFunction.
        
        Args:
            func: The function to convert
            is_setup: Override to mark as setup (for unittest)
            is_teardown: Override to mark as teardown (for unittest)
        """
        # Determine if setup or teardown from decorators if not explicitly set
        if not is_setup and not is_teardown:
            is_setup = any('setup' in d.expression.lower() for d in func.decorators)
            is_teardown = any('teardown' in d.expression.lower() for d in func.decorators)
        
        # Also check method name for unittest conventions
        if not is_setup and not is_teardown:
            method_name_lower = func.name.lower()
            is_setup = 'setup' in method_name_lower
            is_teardown = 'teardown' in method_name_lower
        
        # Get class name if method
        qualified_class_name = ""
        if func.kind == "method":
            # Extract class name from qualified name
            parts = func.qualified_name.rsplit('.', 1)
            if len(parts) > 1:
                qualified_class_name = parts[0]
        
        return FixtureMethod(
            is_setup=is_setup,
            is_teardown=is_teardown,
            qualified_class_name=qualified_class_name,
            qualified_module_name=func.qualified_module_name,
            method_signature=f"{func.name}({', '.join(p.name for p in func.parameters)})",
            method_body="",  # Would need source parsing
            ncloc=self._calculate_ncloc(func),
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(func),
            number_of_constructor_call=self._count_constructor_calls(func),
            number_of_library_call=self._count_library_calls(func),
            number_of_application_call=self._count_application_calls(func),
            number_of_mocking_used=self._count_mocking(func),
        )

    def _analyze_helpers(self, test_func: PyFunction) -> List[HelperMethod]:
        """Analyze helper methods called by the test method.
        
        Only includes helper methods from the same module as the test.
        """
        helpers = []
        
        # Get all application calls
        for call_site in test_func.call_sites:
            if call_site.is_application_call:
                # Try to find the called function
                callee = self._find_callee(call_site)
                if callee and self._is_helper_method(callee, test_func):
                    helper = self._create_helper_method(callee)
                    helpers.append(helper)
                elif callee is None:
                    # If exact match failed, try to find by name in the same module/class
                    callee = self._find_callee_by_name(call_site, test_func)
                    if callee and self._is_helper_method(callee, test_func):
                        helper = self._create_helper_method(callee)
                        helpers.append(helper)
        
        return helpers

    def _find_callee(self, call_site: PyCallSite) -> Optional[PyFunction]:
        """Find the function being called."""
        # Try exact match first
        if call_site.method_name in self.function_lookup:
            return self.function_lookup[call_site.method_name]
        
        # Try qualified name
        qualified_name = f"{call_site.qualified_module_name}.{call_site.method_name}"
        if qualified_name in self.function_lookup:
            return self.function_lookup[qualified_name]
        
        return None

    def _find_callee_by_name(self, call_site: PyCallSite, test_func: PyFunction) -> Optional[PyFunction]:
        """Find a function by name within the same module or class as the test function.
        
        This is a fallback when exact qualified name matching fails.
        Searches for functions with matching names in:
        1. The same module as the test function
        2. The same class as the test function (if test is a method)
        
        Args:
            call_site: The call site to find the callee for
            test_func: The test function making the call
            
        Returns:
            The matching function, or None if not found
        """
        method_name = call_site.method_name
        
        # Get the module containing the test function
        test_module = self._get_module_for_function(test_func)
        if not test_module:
            return None
        
        # Search module-level functions
        for func in test_module.functions:
            if func.name == method_name:
                return func
        
        # If test is a class method, search within the same class
        if test_func.kind == "method":
            # Find the class containing the test method
            for cls in test_module.classes:
                for method in cls.methods:
                    if method.qualified_name == test_func.qualified_name:
                        # Found the test's class, now search for the callee in this class
                        for class_method in cls.methods:
                            if class_method.name == method_name:
                                return class_method
                        break
        
        return None

    def _is_helper_method(self, func: PyFunction, test_func: PyFunction) -> bool:
        """Determine if a function is a helper method.
        
        Helper methods are non-test application methods in the same module as the test.
        
        Args:
            func: The function to check
            test_func: The test function calling this function
            
        Returns:
            True if func is a helper method in the same module
        """
        # Must be non-test
        if func.is_test or func.name.startswith('test_'):
            return False
        
        # Must be in the same module
        if func.qualified_module_name != test_func.qualified_module_name:
            return False
        
        return True

    def _create_helper_method(self, func: PyFunction) -> HelperMethod:
        """Create a HelperMethod from a PyFunction."""
        # Get class name if method
        qualified_class_name = ""
        if func.kind == "method":
            parts = func.qualified_name.rsplit('.', 1)
            if len(parts) > 1:
                qualified_class_name = parts[0]
        
        helper = HelperMethod(
            qualified_class_name=qualified_class_name,
            qualified_module_name=func.qualified_module_name,
            method_signature=f"{func.name}({', '.join(p.name for p in func.parameters)})",
            method_body="",  # Would need source parsing
            ncloc=self._calculate_ncloc(func),
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(func),
            number_of_constructor_call=self._count_constructor_calls(func),
            number_of_library_call=self._count_library_calls(func),
            number_of_application_call=self._count_application_calls(func),
            number_of_mocking_used=self._count_mocking(func),
            assertions=[],
            number_of_assertions=0,
        )
        
        # Analyze assertions in helper
        helper.assertions = self._analyze_assertions(func)
        helper.number_of_assertions = len(helper.assertions)
        
        return helper

    def _analyze_assertions(self, func: PyFunction) -> List[Assertion]:
        """Analyze assertions in a function.
        
        This method detects both:
        1. Assertion method calls (assertEqual, assertTrue, etc.)
        2. Plain assert statements (if tree-sitter is available)
        """
        assertions = []
        
        # 1. Detect assertion method calls from call sites
        for call_site in func.call_sites:
            assertion_type = self._classify_assertion(call_site.method_name)
            if assertion_type:
                assertion = Assertion(
                    assertion_type=assertion_type,
                    assertion_name=call_site.method_name,
                    assertion_code=call_site.method_signature,
                )
                assertions.append(assertion)
        
        # 2. Detect plain assert statements using tree-sitter (if available)
        if ASSERT_DETECTOR_AVAILABLE and self.project_dir:
            module = self._get_module_for_function(func)
            if module and func.start_line > 0 and func.end_line > 0:
                try:
                    file_path = self.project_dir / module.file_path
                    if file_path.exists():
                        assert_statements = detect_assert_statements(
                            str(file_path),
                            func.start_line,
                            func.end_line
                        )
                        assertions.extend(assert_statements)
                except Exception:
                    # If tree-sitter detection fails, continue with method-based assertions
                    pass
        
        return assertions

    def _classify_assertion(self, method_name: str) -> Optional[AssertionType]:
        """Classify an assertion method by its name.
        
        Supports both pytest and unittest assertion methods.
        """
        method_lower = method_name.lower()
        
        # Check if it's an assertion method (pytest or unittest)
        if method_name == 'assert' or method_lower.startswith('assert'):
            # EQUALITY assertions
            if any(keyword in method_lower for keyword in ['equal', 'equals']):
                # unittest: assertEqual, assertEquals, assertNotEqual
                # pytest: assert_equal
                return AssertionType.EQUALITY
            
            # TRUTHINESS assertions
            elif any(keyword in method_lower for keyword in ['true', 'false']):
                # unittest: assertTrue, assertFalse
                # pytest: assert (with boolean)
                return AssertionType.TRUTHINESS
            
            # NULLNESS assertions
            elif any(keyword in method_lower for keyword in ['none', 'null', 'isnone', 'notnone']):
                # unittest: assertIsNone, assertIsNotNone
                # pytest: assert x is None
                return AssertionType.NULLNESS
            
            # IDENTITY assertions
            elif 'is' in method_lower and 'not' not in method_lower:
                # unittest: assertIs, assertIsNot
                # pytest: assert x is y
                return AssertionType.IDENTITY
            
            # THROWABLE assertions
            elif any(keyword in method_lower for keyword in ['raises', 'exception', 'error', 'warns']):
                # unittest: assertRaises, assertRaisesRegex, assertWarns
                # pytest: pytest.raises, assert raises
                return AssertionType.THROWABLE
            
            # COLLECTION assertions
            elif any(keyword in method_lower for keyword in ['in', 'contains', 'count', 'sequence']):
                # unittest: assertIn, assertNotIn, assertCountEqual, assertSequenceEqual
                # pytest: assert x in y
                return AssertionType.COLLECTION
            
            # COMPARISON assertions
            elif any(keyword in method_lower for keyword in ['greater', 'less', 'greaterequal', 'lessequal']):
                # unittest: assertGreater, assertLess, assertGreaterEqual, assertLessEqual
                # pytest: assert x > y
                return AssertionType.COMPARISON
            
            # NUMERIC_TOLERANCE assertions
            elif any(keyword in method_lower for keyword in ['almost', 'close', 'notalmostequal']):
                # unittest: assertAlmostEqual, assertNotAlmostEqual
                # pytest: pytest.approx
                return AssertionType.NUMERIC_TOLERANCE
            
            # TYPE assertions
            elif any(keyword in method_lower for keyword in ['type', 'instance', 'isinstance']):
                # unittest: assertIsInstance, assertNotIsInstance
                # pytest: assert isinstance(x, Type)
                return AssertionType.TYPE
            
            # STRING assertions
            elif any(keyword in method_lower for keyword in ['regex', 'regexp', 'match']):
                # unittest: assertRegex, assertNotRegex, assertRegexpMatches
                return AssertionType.STRING
            
            # Default to TRUTHINESS for generic assert
            else:
                return AssertionType.TRUTHINESS
        
        return None

    def _count_mocking(self, func: PyFunction) -> int:
        """Count mocking usage in a function."""
        mock_count = 0
        
        # Look for mock-related calls
        mock_keywords = ['mock', 'patch', 'spy', 'stub', 'fake', 'monkeypatch']
        
        for call_site in func.call_sites:
            method_lower = call_site.method_name.lower()
            if any(keyword in method_lower for keyword in mock_keywords):
                mock_count += 1
        
        # Look for mock-related decorators
        for decorator in func.decorators:
            decorator_lower = decorator.expression.lower()
            if any(keyword in decorator_lower for keyword in mock_keywords):
                mock_count += 1
        
        return mock_count


def analyze_test_method(
    py_application: PyApplication,
    test_name: str,
    project_dir: Optional[str] = None
) -> Optional[TestMethod]:
    """Convenience function to analyze a test method.
    
    Args:
        py_application: PyApplication model from analysis.json
        test_name: Qualified name of the test method
        project_dir: Optional project directory for source file access
        
    Returns:
        TestMethod model populated with complexity metrics, or None if not found
    """
    analyzer = TestMethodAnalyzer(py_application, project_dir)
    return analyzer.analyze_test_method(test_name)


