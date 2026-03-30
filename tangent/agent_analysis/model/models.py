from typing import List, Any

from pydantic import BaseModel

from tangent.utils.constants import (
    AgenticFramework, TestingFramework, InputType, MockingFramework, MockedResource,
    AssertionType, AgentType, ToolType, MetadataType, OrchestrationPattern,
    MemoryBackend, ErrorHandlingType, AgentPattern
)


class TestInput(BaseModel):
    """Represents information about a test input."""

    method_name: str = ""
    method_signature: str = ""
    receiver_type: str | None = None
    receiver_expr: str | None = None
    input_type: List[InputType] | None = None

class CallableDetails(BaseModel):
    """Represents information about a method/constructor call."""

    method_name: str
    qualified_class_name: str
    qualified_module_name: str
    qualified_method_signature: str
    line_number: int
    column_number: int
    argument_types: List[str] = []
    receiver_type: str | None = None
    secondary_assertion: bool = False
    is_helper: bool = False

class AssertionDetails(BaseModel):
    """Represents information about an assertion."""

    assertion_type: List[AssertionType]
    assertion_name: str
    assertion_code: str | None
    argument_types: List[Any]
    in_helper: bool | None = None
    is_wrapped: bool | None = None

class Fixture(BaseModel):
    is_setup: bool = False
    is_teardown: bool = False
    qualified_class_name: str
    qualified_module_name: str
    method_signature: str
    method_body: str
    ncloc: int = 0
    ncloc_with_helpers: int = 0
    cyclomatic_complexity: int = 0
    cyclomatic_complexity_with_helpers: int = 0
    number_of_objects_created: int = 0
    annotations: List[str] | None = None
    constructor_call_details: List[CallableDetails] | None = None
    application_call_details: List[CallableDetails] | None = None
    library_call_details: List[CallableDetails] | None = None

class Tool(BaseModel):
    """Represents information about a tool used by an agent."""
    framework: AgenticFramework
    tool_name: str
    tool_type: ToolType | None = None
    tool_description: str | None = None
    tool_binding_method: str | None = None
    tool_binding_module: str | None = None

    
class Agent(BaseModel):
    name: str
    agent_type: AgentType
    framework: AgenticFramework
    qualified_class_name: str
    qualified_module_name: str
    method_signature: str
    tools: List[Tool] = []
    description: str = ""
    factory_agent_names: List[str] | None = None  # For CALLER type: names of factory agents being called
    patterns: List[AgentPattern] = []  # Design patterns identified in this agent
    ncloc: int = 0
    ncloc_with_helpers: int = 0
    cyclomatic_complexity: int = 0
    cyclomatic_complexity_with_helpers: int = 0
    number_of_objects_created: int = 0
    annotations: List[str] | None = None
    constructor_call_details: List[CallableDetails] | None = None
    application_call_details: List[CallableDetails] | None = None
    library_call_details: List[CallableDetails] | None = None

class TestMethod(BaseModel):
    agents: List[Agent]
    method_signature: str
    method_declaration: str
    annotations: List[str] | None = None
    thrown_exceptions: List[str] | None = None
    ncloc: int
    ncloc_with_helpers: int = 0
    cyclomatic_complexity: int = 0
    cyclomatic_complexity_with_helpers: int = 0
    test_inputs: List[TestInput] | None = None
    is_mocking_used: bool = False
    number_of_mocks_created: int = 0
    mocking_frameworks_used: List[MockingFramework] | None = None
    mocked_resources: List[MockedResource] | None = None
    number_of_objects_created: int = 0
    number_of_helper_methods: int = 0
    helper_method_ncloc: int = 0
    constructor_call_details: List[CallableDetails] | None = None
    application_call_details: List[CallableDetails] | None = None
    library_call_details: List[CallableDetails] | None = None
    assertion_details: List[AssertionDetails] | None = None


class AgentTest(BaseModel):
    qualified_class_name: str
    qualified_module_name: str
    fixtures: List[Fixture]
    test_methods: List[TestMethod]
    testing_frameworks: List[TestingFramework]

class Application(BaseModel):
    name: str
    framework: List[AgenticFramework]
    tools: List[Tool]
    agents: List[Agent]
    tests: List[AgentTest]