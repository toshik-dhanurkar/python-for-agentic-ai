# Python for Agentic AI — Notes

> Built concept by concept through active recall.
> Each concept answers one practical question about writing production AI code.

---

## Curriculum

1. [Decorators](#concept-1--decorators)
2. [Dataclasses](#concept-2--dataclasses)
3. [Typing Basics](#concept-3--typing-basics)
4. [Enum](#concept-4--enum)
5. [`@property` and `__call__`](#concept-5--property-and-__call__)
6. [Pydantic v2](#concept-6--pydantic-v2)

---

## Concept 1 — Decorators

**Design question:** *How do you add behavior to a function without modifying it?*

### The Problem

Many functions need the same extra behavior — logging, timing, retries, caching. Without decorators you duplicate that logic everywhere:

```python
def login():
    print("[LOG] Starting")
    print("Logging in")

def logout():
    print("[LOG] Starting")
    print("Logging out")
```

A decorator adds behavior once, reuses it everywhere.

---

### First Principle — Functions Are Objects

Functions are first-class objects in Python. They can be assigned to variables, passed as arguments, and returned from other functions:

```python
def greet():
    print("Hello")

x = greet    # assign
x()          # call through the new name
```

This is what makes decorators possible.

---

### Building a Decorator — Without `@`

```python
def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

def greet():
    print("Hello")

greet = decorator(greet)   # manual application
greet()
# Before / Hello / After
```

**Why return a function instead of calling it directly?**

```python
# Wrong:
def decorator(func):
    print("Before")
    func()
    print("After")
    # returns None — greet becomes None
```

The wrapper delays execution until someone actually calls the decorated function. Without it, the decorator runs immediately during decoration and the original function is lost.

---

### What `@` Actually Does

```python
@decorator
def greet():
    print("Hello")
```

is exactly:

```python
def greet():
    print("Hello")

greet = decorator(greet)
```

`@` is syntactic sugar for reassignment. Nothing more.

---

### Generic Decorator — Handling Any Signature

```python
from functools import wraps

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

`*args` and `**kwargs` make the decorator work with any function signature.

---

### Why `functools.wraps`?

Without it, the wrapper replaces the original function's metadata:

```python
@decorator
def greet():
    pass

print(greet.__name__)   # wrapper — wrong!
```

This breaks debugging, stack traces, `help()`, and any framework that inspects function metadata.

`@wraps(func)` copies `__name__`, `__doc__`, `__module__`, `__annotations__` and sets `__wrapped__` so tools can access the original function.

```python
print(greet.__name__)   # greet — correct
```

---

### Decorator Factory — When the Decorator Takes Arguments

```python
@logger("INFO")
def greet():
    pass
```

Requires one extra level of nesting:

```python
def logger(level):           # factory — takes arguments
    def decorator(func):     # actual decorator
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[{level}] {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

Python interprets `@logger("INFO")` as `func = logger("INFO")(func)` — call the factory first, then apply the returned decorator.

---

### Flow

```
Without arguments:
Function → Decorator → Wrapper → Wrapper calls Original

With arguments:
Factory → Decorator → Wrapper → Original
```

---

### Practical Examples

**Logging:**
```python
def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print("Finished")
        return result
    return wrapper
```

**Timing:**
```python
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter()-start:.4f}s")
        return result
    return wrapper
```

---

### Where You'll See Decorators in Agentic AI

```
@tool            → wraps a function as an agent tool
@retry           → retries failed LLM API calls
@cached          → caches expensive embeddings
@agent           → registers a function as an agent
@chain           → links steps in a pipeline
@field_validator → validates Pydantic fields
@abstractmethod  → enforces contracts in base classes
```

Every one of these is just a decorator — a function that takes a function and returns a wrapper.

### One-Liner
> A decorator is a function that takes another function, wraps additional behavior around it, and returns a new function. `@` is just shorthand for `func = decorator(func)`.

---

## Concept 2 — Dataclasses

**Design question:** *How do you eliminate repetitive boilerplate in data-holding classes?*

### The Problem

Classes that primarily store data require the same boilerplate every time:

```python
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"User(name={self.name}, age={self.age})"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age
```

This repeats across dozens of classes. `@dataclass` eliminates all of it.

---

### Basic Dataclass

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
```

Python auto-generates three methods:

| Method | What it does |
|---|---|
| `__init__` | Creates the object with the declared fields |
| `__repr__` | Returns `User(name='Alice', age=24)` |
| `__eq__` | Compares by field values, not object identity |

```python
u1 = User("Alice", 24)
u2 = User("Alice", 24)
print(u1 == u2)   # True  ← __eq__ compares values
print(u1)         # User(name='Alice', age=24)  ← __repr__
```

---

### Default Values

```python
@dataclass
class User:
    name: str
    age: int = 18   # default
```

**Rule:** fields with defaults must come after fields without. This fails:

```python
@dataclass
class User:
    age: int = 18
    name: str        # TypeError — non-default after default
```

---

### The Mutable Default Problem

```python
# Wrong:
@dataclass
class Student:
    subjects: list = []   # shared across ALL instances!
```

The list is created once when the class is defined. Both instances point to the same object:

```python
s1 = Student()
s2 = Student()
s1.subjects.append("Math")
print(s2.subjects)   # ['Math'] — unexpected!
```

**Fix — `field(default_factory=...)`:**

```python
from dataclasses import field

@dataclass
class Student:
    subjects: list = field(default_factory=list)
```

`default_factory=list` calls `list()` separately for every new instance. Each object gets its own list.

---

### `field()` — Customizing Individual Fields

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    age: int
    subjects: list = field(default_factory=list)  # mutable default
    password: str = field(repr=False)              # hide from repr
    id: int = field(compare=False)                 # exclude from __eq__
    score: float = field(init=False)               # not a constructor param
```

| Option | Effect |
|---|---|
| `default_factory` | Calls a function to create a fresh default per instance |
| `repr=False` | Hides field from `__repr__` output |
| `compare=False` | Excludes field from `__eq__` comparison |
| `init=False` | Field not included in `__init__`, set manually or in `__post_init__` |

---

### `__post_init__` — Computed Fields

Runs immediately after the generated `__init__`:

```python
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height

r = Rectangle(4.0, 5.0)
print(r.area)   # 20.0
```

---

### `frozen=True` — Immutable Dataclasses

```python
@dataclass(frozen=True)
class AgentConfig:
    model: str
    temperature: float
    max_tokens: int

cfg = AgentConfig("gpt-5", 0.7, 1000)
cfg.model = "gpt-4"   # FrozenInstanceError
```

Once created, attributes cannot be reassigned. Ideal for configs, coordinates, API settings.

**Critical limitation — shallow only:**

```python
cfg.tools = []           # FrozenInstanceError ✅ blocked
cfg.tools.append("sql")  # Works ❌ list itself is still mutable
```

`frozen=True` also enables hashing — frozen dataclasses can be used as dict keys or in sets.

---

### Dataclass vs Plain Class

| Use `@dataclass` | Use plain class |
|---|---|
| Primarily stores data | Primarily implements behavior |
| Config, schema, result objects | DB connections, game engines, pipelines |

### One-Liner
> A dataclass is a code generator for data-holding classes — you declare the fields, Python generates `__init__`, `__repr__`, and `__eq__`. Use `field()` for mutable defaults and custom behavior, `frozen=True` for immutable config objects.

---

## Concept 3 — Typing Basics

**Design question:** *How do you tell Python (and your tools) what type a variable should be?*

### The Rule — Prefer Modern Syntax

```python
str | None          # not Optional[str]
int | str           # not Union[int, str]
list[str]           # not List[str]
dict[str, int]      # not Dict[str, int]
tuple[int, int]     # not Tuple[int, int]
```

Everything from `typing` that has a built-in equivalent is redundant in Python 3.10+.

---

### Quick Reference

| Type | Meaning | Example |
|---|---|---|
| `X \| None` | Value may be None | `name: str \| None = None` |
| `X \| Y` | One of multiple types | `id: int \| str` |
| `list[X]` | List of one type | `tools: list[str]` |
| `dict[K, V]` | Typed key/value pairs | `scores: dict[str, float]` |
| `tuple[X, Y]` | Fixed-size tuple | `point: tuple[int, int]` |
| `tuple[int, ...]` | Variable-length tuple | `nums: tuple[int, ...]` |
| `Any` | Any type, disables checking | `payload: dict[str, Any]` |
| `Callable[[Args], Return]` | A function with signature | `cb: Callable[[str], bool]` |
| `type[X]` | The class itself, not an instance | `cls: type[Animal]` |
| `Literal[...]` | Only specific values allowed | `mode: Literal["fast", "slow"]` |

---

### Still Need `from typing import`

```python
from typing import Any, Callable, Literal
```

These three have no modern shorthand.

---

### 90% of What You'll Use in Agentic AI

```python
str | None              # optional API keys, responses
list[str]               # tool names, messages
dict[str, Any]          # JSON payloads, metadata
Callable[[str], None]   # callbacks, hooks
type[BaseAgent]         # passing agent classes
Literal["chat", "tool"] # fixed config values
Any                     # escape hatch — use sparingly
```

### One-Liner
> Type hints don't change how Python runs your code — they tell your editor, linter, and Pydantic what to expect. Use modern `X | Y` and `list[X]` syntax in all new code.

---

## Concept 4 — Enum

**Design question:** *How do you define a fixed set of allowed values that can't be accidentally mistyped at runtime?*

### The Problem

```python
run_agent("Chat")   # typo — no error, wrong behavior
run_agent("chatt")  # no error, silently does nothing
```

Magic strings and magic numbers break silently at runtime. Enums eliminate both.

---

### Basic Enum

```python
from enum import Enum

class AgentMode(Enum):
    CHAT = "chat"
    TOOL = "tool"
    PLANNER = "planner"

mode = AgentMode.CHAT
print(mode.name)    # CHAT
print(mode.value)   # chat
```

---

### IntEnum, `auto()`, Iteration, Lookup

```python
from enum import IntEnum, auto

class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

Priority.HIGH > Priority.LOW   # True — members compare like ints
Priority.HIGH == 3             # True

class Status(Enum):
    PENDING = auto()   # 1
    RUNNING = auto()   # 2
    DONE = auto()      # 3

# Iterate
for mode in AgentMode:
    print(mode.name, mode.value)

# Lookup by value
AgentMode("chat")    # AgentMode.CHAT
AgentMode("unknown") # ValueError

# Lookup by name
AgentMode["CHAT"]    # AgentMode.CHAT
```

---

### Enum vs Literal

| | `Enum` | `Literal` |
|---|---|---|
| Shared across many modules | ✅ | ❌ |
| Needs iteration / `.name` / `.value` | ✅ | ❌ |
| Runtime validation | ✅ | ❌ |
| Just a type hint in one place | ❌ | ✅ |

### One-Liner
> Enum creates a named set of constant objects — safe, autocomplete-friendly, runtime-validatable. Use `Enum` for shared constants, `IntEnum` for numeric codes, `auto()` when values don't matter, `Literal` when you only need a type hint in one place.

---

## Concept 5 — `@property` and `__call__`

**Design question:** *How do you add validation to attribute access without breaking existing code? And how do you make an object behave like a function?*

---

### `@property`

```python
class User:
    def __init__(self, age):
        self._age = age

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

    @age.deleter
    def age(self):
        del self._age

u = User(20)
print(u.age)   # getter
u.age = 25     # setter — runs validation
del u.age      # deleter
```

Attribute syntax stays the same. No callers need to change. Validation added transparently.

**Computed properties:**
```python
class Rectangle:
    def __init__(self, w, h):
        self.width = w
        self.height = h

    @property
    def area(self):
        return self.width * self.height

r = Rectangle(4, 5)
print(r.area)   # not r.area() — looks like an attribute
```

**When to use:** validate on assignment, compute values on demand, keep a stable public API while changing internals.

**Mental model:** `u.age` → Python sees property descriptor → calls getter method via `User.age.__get__(u, User)`.

---

### `__call__`

```python
class SimilarityScorer:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, score):
        return score >= self.threshold

scorer = SimilarityScorer(0.8)
scorer(0.92)   # True — called like a function, but carries state
scorer(0.65)   # False
```

**Agentic AI pattern:**
```python
class SearchTool:
    def __init__(self, api_client):
        self.client = api_client

    def __call__(self, query: str) -> str:
        return self.client.search(query)

search = SearchTool(client)
result = search("Python decorators")   # function-like interface, object internals
```

**Rule:** stateless logic → regular function. Stateful behavior → `__call__`.

| Feature | Purpose |
|---|---|
| `@property` | Method that reads/writes like an attribute |
| `@x.setter` | Validates on assignment |
| `__call__` | Makes an object callable |

### One-Liner
> `@property` hides a method behind attribute syntax. `__call__` makes an object behave like a function — use it when a function needs to carry state.

---

## Concept 6 — Pydantic v2

**Design question:** *How do you safely convert untrusted external data into validated Python objects?*

### Why Pydantic — Not Just a Better Dataclass

Dataclasses trust you:

```python
@dataclass
class AgentConfig:
    model: str
    temperature: float

cfg = AgentConfig("gpt-4", "0.8")
print(cfg.temperature)   # "0.8" — still a string, no error
```

Pydantic verifies incoming data:

```python
from pydantic import BaseModel

class AgentConfig(BaseModel):
    model: str
    temperature: float

cfg = AgentConfig(model="gpt-4", temperature="0.8")
print(cfg.temperature)        # 0.8 — converted to float
print(type(cfg.temperature))  # <class 'float'>
```

```python
AgentConfig(model="gpt-4", temperature="high")  # ValidationError — can't convert
```

| Dataclass | Pydantic |
|---|---|
| Stores data | Validates + converts data |
| Trusts input | Checks input |
| No JSON support | model_validate_json() built-in |
| No detailed errors | Rich ValidationError with field-level messages |

**The mental model:**
```
Raw Data (JSON, API response, env vars)
      ↓
Pydantic (BaseModel + types + Field + validators)
      ↓
Trusted Python objects
      ↓
Business / Agent logic
      ↓
model_dump() / model_dump_json()
      ↓
Back to JSON, APIs, tool calls
```

Pydantic is the **boundary between the messy outside world and your clean Python objects**.

---

### BaseModel — Creating and Using Models

```python
from pydantic import BaseModel

class LLMResponse(BaseModel):
    answer: str
    tokens_used: int
    latency: float

raw = {"answer": "Paris", "tokens_used": "42", "latency": "1.2"}
response = LLMResponse(**raw)   # strings converted automatically

print(response.answer)       # Paris
print(response.tokens_used)  # 42  (int, not "42")
```

---

### Type Hints Become Runtime Validation

Python normally ignores type hints at runtime. Pydantic reads them as a schema:

```python
# Python: add("hello", "world") works — type hints are ignored
# Pydantic: same annotation means "validate and convert at construction time"
```

The validation pipeline:
```
Incoming Data → Read annotations → Convert types → Apply Field constraints
             → Run field validators → Run model validators → Construct object
```

---

### `Field()` — Constraints, Defaults, Aliases, Descriptions

Type hints answer "what is this?" — `Field()` answers "what rules apply?"

```python
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    model: str = Field(default="gpt-4", min_length=3)
    temperature: float = Field(ge=0, le=2, description="Sampling temperature")
    max_tokens: int = Field(gt=0, le=4096)
    retries: int = Field(default=3, ge=1, le=10)
```

**Common constraints:**

| Constraint | Meaning |
|---|---|
| `gt` / `ge` | greater than / greater than or equal |
| `lt` / `le` | less than / less than or equal |
| `min_length` / `max_length` | string/list length |
| `pattern` | regex pattern for strings |
| `description` | used by FastAPI/tool schemas |

**Aliases — when JSON uses camelCase:**

```python
class Tool(BaseModel):
    tool_name: str = Field(alias="toolName")

Tool(toolName="search")   # works — alias maps to tool_name
```

---

### Nested Models

```python
class Tool(BaseModel):
    name: str
    description: str

class Agent(BaseModel):
    name: str
    tool: Tool   # nested model

agent = Agent(
    name="Research Agent",
    tool={"name": "search", "description": "Search the web"}  # dict auto-converted
)

print(agent.tool.name)   # search
```

Pydantic automatically converts the dict to a `Tool` instance. Validation is recursive.

---

### Lists of Models

```python
class Agent(BaseModel):
    name: str
    tools: list[Tool]   # list of nested models

agent = Agent(
    name="Research Agent",
    tools=[
        {"name": "search", "description": "Search"},
        {"name": "calculator", "description": "Math"},
    ]
)

print(agent.tools[0].name)   # search
```

Each dict in the list is automatically converted to a `Tool`.

---

### `model_dump()` and `model_dump_json()`

```python
config = AgentConfig(model="gpt-5", temperature=0.7, max_tokens=1000, retries=3)

config.model_dump()
# {"model": "gpt-5", "temperature": 0.7, "max_tokens": 1000, "retries": 3}

config.model_dump_json()
# '{"model":"gpt-5","temperature":0.7,"max_tokens":1000,"retries":3}'
```

Nested models are serialized recursively.

---

### `model_validate()` and `model_validate_json()`

```python
# From dict (preferred over **unpacking in v2)
raw = {"model": "gpt-5", "temperature": "0.7", "max_tokens": "1000", "retries": 3}
config = AgentConfig.model_validate(raw)

# From JSON string (LLM output, API response)
json_str = '{"model": "gpt-5", "temperature": 0.7, "max_tokens": 1000, "retries": 3}'
config = AgentConfig.model_validate_json(json_str)  # no json.loads() needed
```

**The complete lifecycle:**
```
LLM JSON response
      ↓
model_validate_json()     ← parsing/ingestion (outside → inside)
      ↓
Validated BaseModel
      ↓
Agent / Tool Logic
      ↓
model_dump()              ← serialization/export (inside → outside)
      ↓
dict / JSON sent to next tool or API
```

---

### `@field_validator` — Single Field Custom Rules

When type constraints aren't enough:

```python
from pydantic import BaseModel, Field, field_validator

class Tool(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(default="No description")
    max_retries: int = Field(ge=1, le=5)

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, value: str) -> str:
        return value.upper()   # transform before storing

tool = Tool(name="search", max_retries=3)
print(tool.name)   # SEARCH
```

```python
class Config(BaseModel):
    model: str

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        allowed = {"gpt-4", "gpt-5", "claude"}
        if value not in allowed:
            raise ValueError(f"Unknown model: {value}")
        return value
```

Field validators run **after** type validation. They receive the already-converted value.

---

### `@model_validator` — Cross-Field Validation

When a rule involves two or more fields together:

```python
from pydantic import BaseModel, model_validator

class LLMConfig(BaseModel):
    model: str
    temperature: float
    use_json_mode: bool

    @model_validator(mode="after")
    def validate_o1_temperature(self):
        if self.model == "o1" and self.temperature != 1.0:
            raise ValueError("o1 only supports temperature=1.0")
        return self
```

```python
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict

    @model_validator(mode="after")
    def validate_calculator_args(self):
        if self.tool_name == "calculator" and "expression" not in self.arguments:
            raise ValueError("Calculator requires 'expression' in arguments")
        return self
```

`mode="after"` means the validator runs after all field-level validation — `self` already has the fully validated fields.

---

### Complete AI Engineering Example

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class Tool(BaseModel):
    name: str
    description: str

class AgentConfig(BaseModel):
    model: str = Field(min_length=2)
    temperature: float = Field(ge=0, le=2)
    tools: list[Tool]

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        allowed = {"gpt-4", "gpt-5", "claude"}
        if value not in allowed:
            raise ValueError(f"Unsupported model: {value}")
        return value

    @model_validator(mode="after")
    def validate_tools_not_empty(self):
        if not self.tools:
            raise ValueError("Agent needs at least one tool")
        return self

# Load from raw JSON (e.g. LLM output or config file)
config = AgentConfig.model_validate({
    "model": "gpt-5",
    "temperature": "0.7",   # string → float automatically
    "tools": [
        {"name": "search", "description": "Search the web"},
        {"name": "calculator", "description": "Solve math"},
    ]
})

print(config.model_dump())
print(config.model_dump_json())
```

---

### One-Liner
> Pydantic is the boundary between untrusted external data and your clean Python objects. Type hints become a schema, `Field()` adds constraints and metadata, `@field_validator` adds single-field business rules, `@model_validator` adds cross-field rules. `model_validate()` parses in, `model_dump()` serializes out.

---

## Master Summary

- A **decorator** is a function that takes a function and returns a new function. `@decorator` is shorthand for `func = decorator(func)`.
- Always use `functools.wraps` — without it, `__name__`, `__doc__`, and all metadata are replaced by the wrapper's.
- Decorator factories add one extra level: `@logger("INFO")` → `logger("INFO")` returns a decorator, which wraps the function.
- **`@dataclass`** auto-generates `__init__`, `__repr__`, and `__eq__` for data-holding classes.
- Never use mutable objects (`list`, `dict`) as direct defaults — use `field(default_factory=list)` instead.
- `field()` customizes individual fields: `default_factory`, `repr=False`, `compare=False`, `init=False`.
- `__post_init__` runs after `__init__` — use it for computed fields.
- `frozen=True` prevents attribute reassignment and enables hashing, but is shallow — mutable objects inside are still mutable.
- Use `@dataclass` for data containers (configs, results, messages). Use plain classes for behavior-heavy objects.
- **Typing**: prefer modern syntax — `str | None`, `list[str]`, `dict[str, Any]`. Only `Any`, `Callable`, `Literal` still need `from typing import`.
- Type hints don't affect runtime — they exist for editors, linters, and Pydantic validation.
- **Enum** creates a named set of constant objects. `IntEnum` for numeric codes. `auto()` when values don't matter. `Literal` when you just need a type hint in one place.
- **`@property`** hides a method behind attribute syntax — add validation without breaking callers. `__call__` makes an object behave like a function — use it when a function needs to carry state.
- **Pydantic `BaseModel`**: type hints become runtime validation. Strings are coerced, invalid data raises `ValidationError`.
- `Field()` adds constraints (`ge`, `le`, `min_length`), defaults, aliases, and descriptions beyond what type hints can express.
- Nested models and `list[Model]` are validated recursively — pass dicts, get validated objects.
- `model_validate()` = parsing in (external data → object). `model_dump()` = serialization out (object → dict/JSON).
- `@field_validator` validates one field with custom logic. `@model_validator(mode="after")` validates cross-field relationships after all fields are set.
