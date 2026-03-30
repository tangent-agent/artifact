from typing import List

from pydantic import BaseModel

from tangent.utils.constants import AssertionType


class Assertion(BaseModel):
    assertion_type: AssertionType
    assertion_name: str = ""
    assertion_code: str | None = None
    
class FixtureMethod(BaseModel):
    is_setup: bool = False
    is_teardown: bool = False
    qualified_class_name: str
    qualified_module_name: str
    method_signature: str
    method_body: str
    ncloc: int = 0
    number_of_mocking_used: int = 0
    # Complexity metrics
    ncloc: int
    cyclomatic_complexity: int = 0
    # Call counts
    number_of_constructor_call: int
    number_of_library_call: int
    number_of_application_call: int

class HelperMethod(BaseModel):
    qualified_class_name: str
    qualified_module_name: str
    method_signature: str
    method_body: str
    ncloc: int = 0
    number_of_mocking_used: int = 0
    # Complexity metrics
    ncloc: int
    cyclomatic_complexity: int = 0
    # Call counts
    number_of_constructor_call: int
    number_of_library_call: int
    number_of_application_call: int
    assertions: List[Assertion] = []
    number_of_assertions: int = 0

class TestMethod(BaseModel):
    """Represents a test method with complexity metrics."""
    
    test_name: str
    qualified_module_name: str
    file_path: str
    start_line: int = -1
    end_line: int = -1

    # fixtures
    number_of_fixtures_used: int = 0
    fixtures: List[FixtureMethod]

    # helper methods
    number_of_helper_methods: int = 0
    helpers: List[HelperMethod]

    # Mocking
    number_of_mocking_used: int = 0

    # Complexity metrics
    ncloc: int
    cyclomatic_complexity: int = 0
    
    # Call counts
    number_of_constructor_call: int
    number_of_library_call: int
    number_of_application_call: int
    
    # Assertions
    assertions: List[Assertion] = []
    number_of_assertions: int = 0
    
    # Async flag
    is_async: bool = False
    
