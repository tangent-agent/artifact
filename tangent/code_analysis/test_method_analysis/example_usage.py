"""Example usage of the test method analyzer.

This script demonstrates how to use the test method analyzer to populate
the TestMethod model from a PyApplication and test name.
"""

import json
from pathlib import Path
from typing import Optional

from tangent.code_analysis.model.model import PyApplication
from tangent.code_analysis.test_method_analysis import analyze_test_method


def load_analysis_json(analysis_path: str) -> PyApplication:
    """Load PyApplication from analysis.json file.
    
    Args:
        analysis_path: Path to the analysis.json file
        
    Returns:
        PyApplication model
    """
    with open(analysis_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return PyApplication(**data)


def analyze_and_print_test(
    analysis_path: str,
    test_name: str,
    project_dir: Optional[str] = None
):
    """Analyze a test method and print the results.
    
    Args:
        analysis_path: Path to the analysis.json file
        test_name: Qualified name of the test method
        project_dir: Optional project directory for source file access
    """
    # Load the PyApplication model
    print(f"Loading analysis from: {analysis_path}")
    py_app = load_analysis_json(analysis_path)
    
    # Analyze the test method
    print(f"\nAnalyzing test method: {test_name}")
    test_method = analyze_test_method(py_app, test_name, project_dir)
    
    if test_method is None:
        print(f"Error: Test method '{test_name}' not found in analysis")
        return
    
    # Print the results
    print("\n" + "="*80)
    print("TEST METHOD ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nTest Name: {test_method.test_name}")
    print(f"Module: {test_method.qualified_module_name}")
    print(f"File: {test_method.file_path}")
    print(f"Lines: {test_method.start_line}-{test_method.end_line}")
    
    print(f"\n--- Complexity Metrics ---")
    print(f"NCLOC: {test_method.ncloc}")
    print(f"Cyclomatic Complexity: {test_method.cyclomatic_complexity}")
    
    print(f"\n--- Call Counts ---")
    print(f"Constructor Calls: {test_method.number_of_constructor_call}")
    print(f"Library Calls: {test_method.number_of_library_call}")
    print(f"Application Calls: {test_method.number_of_application_call}")
    
    print(f"\n--- Test Components ---")
    print(f"Fixtures Used: {test_method.number_of_fixtures_used}")
    for i, fixture in enumerate(test_method.fixtures, 1):
        print(f"  {i}. {fixture.method_signature}")
        print(f"     - Setup: {fixture.is_setup}, Teardown: {fixture.is_teardown}")
        print(f"     - NCLOC: {fixture.ncloc}, Complexity: {fixture.cyclomatic_complexity}")
    
    print(f"\nHelper Methods: {test_method.number_of_helper_methods}")
    for i, helper in enumerate(test_method.helpers, 1):
        print(f"  {i}. {helper.method_signature}")
        print(f"     - NCLOC: {helper.ncloc}, Complexity: {helper.cyclomatic_complexity}")
        print(f"     - Assertions: {helper.number_of_assertions}")
    
    print(f"\nMocking Used: {test_method.number_of_mocking_used}")
    
    print(f"\nAssertions: {test_method.number_of_assertions}")
    for i, assertion in enumerate(test_method.assertions, 1):
        print(f"  {i}. {assertion.assertion_name} ({assertion.assertion_type.value})")
    
    print("\n" + "="*80)
    
    # Export to JSON
    print("\nExporting to JSON...")
    output = test_method.model_dump_json(indent=2)
    output_path = f"test_analysis_{test_method.test_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"Saved to: {output_path}")


def main():
    """Main entry point for example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python example_usage.py <analysis.json> <test_name> [project_dir]")
        print("\nExample:")
        print("  python example_usage.py cldk_cache/analysis.json test_module.TestClass.test_method")
        print("  python example_usage.py cldk_cache/analysis.json test_module.TestClass.test_method /path/to/project")
        sys.exit(1)
    
    analysis_path = sys.argv[1]
    test_name = sys.argv[2]
    project_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyze_and_print_test(analysis_path, test_name, project_dir)


if __name__ == "__main__":
    main()


