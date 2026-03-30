"""Test method analysis module.

This module provides functionality to analyze test methods and extract
complexity metrics including fixtures, helpers, assertions, and mocking usage.

Supports both method-based assertions (assertEqual, assertTrue, etc.) and
plain assert statements (via tree-sitter if available).
"""

from tangent.code_analysis.test_method_analysis.test_analyzer import (
    TestMethodAnalyzer,
    analyze_test_method,
    ASSERT_DETECTOR_AVAILABLE,
)

# Try to export assert detector if available
try:
    from tangent.code_analysis.test_method_analysis.assert_detector import (
        AssertStatementDetector,
        detect_assert_statements,
    )
    __all__ = [
        "TestMethodAnalyzer",
        "analyze_test_method",
        "AssertStatementDetector",
        "detect_assert_statements",
        "ASSERT_DETECTOR_AVAILABLE",
    ]
except ImportError:
    __all__ = [
        "TestMethodAnalyzer",
        "analyze_test_method",
        "ASSERT_DETECTOR_AVAILABLE",
    ]


