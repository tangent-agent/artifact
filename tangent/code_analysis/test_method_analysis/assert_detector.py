"""Tree-sitter based assertion detector for plain assert statements.

This module uses tree-sitter to parse Python source code and detect
plain assert statements that are not captured by call-site analysis.
"""

import re
from pathlib import Path
from typing import List, Optional

try:
    from tree_sitter import Language, Parser, Node
    import tree_sitter_python as tspython
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None
    Node = None

from tangent.code_analysis.model.test_complexity_model import Assertion
from tangent.utils.constants import AssertionType


class AssertStatementDetector:
    """Detects plain assert statements using tree-sitter."""
    
    def __init__(self):
        """Initialize the tree-sitter parser."""
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter is not available. Install with: "
                "pip install tree-sitter tree-sitter-python"
            )
        
        # Initialize Python parser with the new API (tree-sitter 0.25+)
        # tspython.language() returns a PyCapsule that needs to be wrapped
        py_language = Language(tspython.language())
        self.parser = Parser(py_language)
    
    def detect_assertions_in_source(
        self,
        source_code: str,
        start_line: int,
        end_line: int
    ) -> List[Assertion]:
        """Detect assert statements in source code within line range.
        
        Args:
            source_code: Full source code of the file
            start_line: Start line of the function (1-based)
            end_line: End line of the function (1-based)
            
        Returns:
            List of Assertion objects for detected assert statements
        """
        if not TREE_SITTER_AVAILABLE:
            return []
        
        # Parse the source code
        tree = self.parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        
        # Find assert statements within the line range
        assertions = []
        self._find_asserts_in_range(
            root_node,
            start_line,
            end_line,
            source_code,
            assertions
        )
        
        return assertions
    
    def _find_asserts_in_range(
        self,
        node: 'Node',
        start_line: int,
        end_line: int,
        source_code: str,
        assertions: List[Assertion]
    ):
        """Recursively find assert statements in the given line range."""
        # Check if this node is an assert statement
        if node.type == 'assert_statement':
            # Get the line number (tree-sitter uses 0-based indexing)
            assert_line = node.start_point[0] + 1
            
            # Check if it's within our range
            if start_line <= assert_line <= end_line:
                # Extract the assertion code
                assert_code = source_code[node.start_byte:node.end_byte]
                
                # Classify the assertion
                assertion_type = self._classify_assert_statement(node, source_code)
                
                assertion = Assertion(
                    assertion_type=assertion_type,
                    assertion_name="assert",
                    assertion_code=assert_code
                )
                assertions.append(assertion)
        
        # Recursively check children
        for child in node.children:
            self._find_asserts_in_range(
                child,
                start_line,
                end_line,
                source_code,
                assertions
            )
    
    def _classify_assert_statement(self, node: 'Node', source_code: str) -> AssertionType:
        """Classify an assert statement by analyzing its structure.
        
        Args:
            node: The assert_statement node
            source_code: Full source code
            
        Returns:
            AssertionType enum value
        """
        # Get the assertion code
        assert_code = source_code[node.start_byte:node.end_byte].lower()
        
        # Get the test expression (first child after 'assert' keyword)
        test_expr = None
        for child in node.children:
            if child.type != 'assert':
                test_expr = child
                break
        
        if not test_expr:
            return AssertionType.TRUTHINESS
        
        # Analyze the test expression structure
        test_code = source_code[test_expr.start_byte:test_expr.end_byte].lower()
        
        # Check for specific patterns
        
        # NULLNESS: assert x is None, assert x is not None
        if ' is none' in test_code or ' is not none' in test_code:
            return AssertionType.NULLNESS
        
        # IDENTITY: assert x is y, assert x is not y
        if ' is ' in test_code and ' is none' not in test_code:
            return AssertionType.IDENTITY
        
        # EQUALITY: assert x == y, assert x != y
        if '==' in test_code or '!=' in test_code:
            return AssertionType.EQUALITY
        
        # COMPARISON: assert x > y, assert x < y, assert x >= y, assert x <= y
        if any(op in test_code for op in ['>=', '<=', '>', '<']):
            return AssertionType.COMPARISON
        
        # COLLECTION: assert x in y, assert x not in y
        if ' in ' in test_code:
            return AssertionType.COLLECTION
        
        # NUMERIC_TOLERANCE: assert isclose(...), assert np.isclose(...), assert math.isclose(...)
        if 'isclose' in test_code or 'allclose' in test_code:
            return AssertionType.NUMERIC_TOLERANCE
        
        # TYPE: assert isinstance(...), assert issubclass(...)
        if 'isinstance' in test_code or 'issubclass' in test_code or 'type(' in test_code:
            return AssertionType.TYPE
        
        # STRING: assert x.startswith(...), assert x.endswith(...), assert x in string
        if any(method in test_code for method in ['startswith', 'endswith', 'match', 'search']):
            return AssertionType.STRING
        
        # TRUTHINESS: Default for boolean expressions
        return AssertionType.TRUTHINESS
    
    def detect_assertions_in_file(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> List[Assertion]:
        """Detect assert statements in a file within line range.
        
        Args:
            file_path: Path to the Python file
            start_line: Start line of the function (1-based)
            end_line: End line of the function (1-based)
            
        Returns:
            List of Assertion objects for detected assert statements
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            return self.detect_assertions_in_source(source_code, start_line, end_line)
        except Exception as e:
            # If we can't read the file or parse it, return empty list
            return []


def detect_assert_statements(
    file_path: str,
    start_line: int,
    end_line: int
) -> List[Assertion]:
    """Convenience function to detect assert statements.
    
    Args:
        file_path: Path to the Python file
        start_line: Start line of the function (1-based)
        end_line: End line of the function (1-based)
        
    Returns:
        List of Assertion objects, or empty list if tree-sitter not available
    """
    if not TREE_SITTER_AVAILABLE:
        return []
    
    try:
        detector = AssertStatementDetector()
        return detector.detect_assertions_in_file(file_path, start_line, end_line)
    except Exception:
        return []


