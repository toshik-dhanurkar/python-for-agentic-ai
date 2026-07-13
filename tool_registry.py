from enum import Enum
from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator, model_validator
from pydantic import BaseModel
import json


# ==========================================================
# 1. ToolType
# ==========================================================

class ToolType(str, Enum):
    SEARCH = "search"
    CALCULATOR = "calculator"
    API_CALL = "api_call"


# ==========================================================
# 2. ToolConfig
# ==========================================================

class ToolConfig(BaseModel):
    timeout: int = Field(ge=1, le=60)
    max_retries: int = Field(default=3, ge=0, le=5)
    api_key: str | None = None

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith("sk-"):
            raise ValueError("API key must start with 'sk-'")
        return value


# ==========================================================
# 3. Tool
# ==========================================================

class Tool(BaseModel):
    name: str = Field(min_length=2)
    tool_type: ToolType
    description: str = "No description"
    config: ToolConfig
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def lowercase_name(cls, value: str) -> str:
        return value.lower()

    @computed_field
    @property
    def summary(self) -> str:
        return f"{self.name} ({self.tool_type.value}) - {self.description}"


# ==========================================================
# 4. ToolExecutor
# ==========================================================

class ToolExecutor:

    def __init__(self, tool: Tool):
        self.tool = tool
        self._execution_count = 0

    def __call__(self, input_data: str) -> str:
        if not self.tool.enabled:
            raise RuntimeError(f"Tool '{self.tool.name}' is disabled.")
        self._execution_count += 1
        return f"[{self.tool.name}] Executed with: {input_data}"

    @property
    def execution_count(self) -> int:
        return self._execution_count


# ==========================================================
# 5. ToolRegistry
# ==========================================================

class ToolRegistry:

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._executors: dict[str, ToolExecutor] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self.tools[tool.name] = tool
        self._executors[tool.name] = ToolExecutor(tool)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found.")
        return self.tools[name]

    def disable(self, name: str) -> None:
        self.get(name).enabled = False

    def enable(self, name: str) -> None:
        self.get(name).enabled = True

    def execute(self, name: str, input_data: str) -> str:
        tool = self.get(name)
        if not tool.enabled:
            raise RuntimeError(f"Tool '{name}' is disabled.")
        executor = self._executors[name]
        return executor(input_data)

    def execution_count(self, name: str) -> int:
        if name not in self._executors:
            raise KeyError(f"Tool '{name}' not found.")
        return self._executors[name].execution_count

    def list_enabled(self) -> list[Tool]:
        return [t for t in self.tools.values() if t.enabled]

    def to_schema(self) -> list[dict]:
        return [tool.model_dump() for tool in self.tools.values()]

    @classmethod
    def from_config(cls, data: list[dict]) -> "ToolRegistry":
        registry = cls()
        for item in data:
            tool = Tool.model_validate(item)
            registry.register(tool)
        return registry


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":

    raw_tools = [
        {
            "name": "Search",
            "tool_type": "search",
            "description": "Search the web",
            "config": {"timeout": 20, "api_key": "sk-search"}
        },
        {
            "name": "Calculator",
            "tool_type": "calculator",
            "description": "Perform calculations",
            "config": {"timeout": 10}
        },
        {
            "name": "WeatherAPI",
            "tool_type": "api_call",
            "description": "Weather service",
            "config": {"timeout": 30, "max_retries": 5, "api_key": "sk-weather"}
        }
    ]

    registry = ToolRegistry.from_config(raw_tools)

    # 1. All enabled tools
    print("Enabled Tools")
    print("-" * 40)
    for tool in registry.list_enabled():
        print(tool.summary)

    # 2. Execute a tool
    print("\nExecute Tool")
    print("-" * 40)
    print(registry.execute("search", "Latest AI news"))
    print(registry.execute("search", "Pydantic v2 docs"))
    print(f"search execution count: {registry.execution_count('search')}")

    # 3. Disable a tool
    print("\nDisable Search")
    print("-" * 40)
    registry.disable("search")
    for tool in registry.list_enabled():
        print(tool.summary)

    # 4. Schema output
    print("\nSchema")
    print("-" * 40)
    from pprint import pprint
    pprint(registry.to_schema())

    # 5. Validation errors
    print("\nValidation Errors")
    print("-" * 40)

    try:
        ToolConfig(timeout=100)
    except ValidationError as e:
        print("timeout=100:", e.errors()[0]["msg"])

    try:
        ToolConfig(timeout=10, api_key="bad_key")
    except ValidationError as e:
        print("bad api_key:", e.errors()[0]["msg"])

    try:
        Tool(name="x", tool_type="search", config={"timeout": 5})
    except ValidationError as e:
        print("name too short:", e.errors()[0]["msg"])
