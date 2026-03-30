from enum import Enum

ANALYSIS_FOLDER = 'cldk_cache'
CODE_ANALYSIS_FILENAME = 'analysis.json'
AGENT_ANALYSIS_FILENAME = 'agent_analysis.json'
class AgenticFramework(Enum):
    LangChain = "langchain"
    LangGraph = "langgraph"
    CrewAI = "crewai"
    AutoGen = "autogen"
    SemanticKernel = "semantic_kernel"
    LlamaIndex = "llama_index"
    Haystack = "haystack"
    MetaGPT = "metagpt"
    OpenAISwarm = "swarm"
    OpenAIAgentsSDK = "openai_agents_sdk"
    DSPy = "dspy"
    Langroid = "langroid"
    CAMEL = "camel"
    Letta = "letta"
    MCP = "mcp"
    FastMCP = "fastmcp"
    ClaudeAgentSDK = "claude_agent_sdk"
    Unknown = "unknown"

class TestingFramework(Enum):
    PyUnit = "pyunit"

class InputType(Enum):
    PROPERTIES = "properties"  # java.util.Properties
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    SQL = "sql"
    CSV = "csv"
    XLS = "xls"
    HTML = "html"
    BINARY = "binary"
    SERIALIZED = "serialized"
    RESOURCE = "resource"  # Classpath-based resources

class MockingFramework(Enum):
    MOCKITO = "mockito"
    EASY_MOCK = "easy-mock"
    POWER_MOCK = "power-mock"
    WIRE_MOCK = "wire-mock"
    MOCK_SERVER = "mock-server"
    SPRING_TEST = "spring-test"
    JAVA_REFLECTION = "java-reflection"

class MockedResource(Enum):
    DB = "db"
    FILE = "file"
    APPLICATION_CLASS = "application-class"
    LIBRARY_CLASS = "library-class"
    SERVICE = "service"

class AssertionType(Enum):
    TRUTHINESS = "truthiness"
    EQUALITY = "equality"
    IDENTITY = "identity"
    NULLNESS = "nullness"
    NUMERIC_TOLERANCE = "numeric-tolerance"
    THROWABLE = "throwable"
    TIMEOUT = "timeout"
    COLLECTION = "collection"
    STRING = "string"
    COMPARISON = "comparison"
    TYPE = "type"
    GROUPED = "grouped"
    PROPERTY = "property"
    WRAPPER = "wrapper"  # For assertThat
    UTILITY = "utility"  # For things like assigning assertion returns ("as" in AssertJ)

class AgentType(Enum):
    FACTORY = "factory"  # Agent definition/factory function that returns an agent
    INSTANTIATION = "instantiation"  # Agent creation/usage (creates but doesn't return)
    CALLER = "caller"  # Method that calls a factory agent method
    ORCHESTRATOR = "orchestrator"  # Agent that coordinates and manages other agents
    MEMORY = "memory"  # Agent that handles memory and state management
    TOOL = "tool"  # Agent that primarily provides tool functionality

class ToolType(Enum):
    DECORATOR_TOOL = "decorator_tool"
    CLASS_TOOL = "class_tool"
    FUNCTION_TOOL = "function_tool"
    LAMBDA_TOOL = "lambda_tool"
    MCP_TOOL = "mcp_tool"
    UNKNOWN = "unknown"

class MetadataType(Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER_PROMPT = "user_prompt"
    INSTRUCTIONS = "instructions"
    LLM_CONFIG = "llm_config"

class OrchestrationPattern(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HIERARCHICAL = "hierarchical"
    UNKNOWN = "unknown"

class MemoryBackend(Enum):
    CHROMA = "chroma"
    PINECONE = "pinecone"
    FAISS = "faiss"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    REDIS = "redis"
    POSTGRES = "postgres"
    MONGODB = "mongodb"
    UNKNOWN = "unknown"

class ErrorHandlingType(Enum):
    TRY_EXCEPT = "try_except"
    DECORATOR_RETRY = "decorator_retry"
    MANUAL_RETRY_LOOP = "manual_retry_loop"
    FOR_LOOP_RETRY = "for_loop_retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMITING = "rate_limiting"
    TIMEOUT = "timeout"
    NONE = "none"

class AgentPattern(Enum):
    # Coordination Patterns
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    
    # Communication Patterns
    MESSAGE_PASSING = "message_passing"
    PUB_SUB = "pub_sub"
    REQUEST_RESPONSE = "request_response"
    STREAMING = "streaming"
    
    # State Management
    STATEFUL = "stateful"
    STATELESS = "stateless"
    SHARED_STATE = "shared_state"
    EVENT_SOURCING = "event_sourcing"
    
    # Prompt Engineering
    REACT = "react"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    SELF_CONSISTENCY = "self_consistency"
    REFLECTION = "reflection"
    
    # Tool Usage
    TOOL_CALLING = "tool_calling"
    FUNCTION_CALLING = "function_calling"
    MULTI_TOOL = "multi_tool"
    
    # Specialization
    DOMAIN_SPECIFIC = "domain_specific"
    GENERAL_PURPOSE = "general_purpose"
    ROUTER = "router"
    
    # Other
    SIMPLE = "simple"
    UNKNOWN = "unknown"