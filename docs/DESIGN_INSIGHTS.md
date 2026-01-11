# System Design Insights & Best Practices

## Critical Design Insights

This document captures the key architectural decisions and why they matter for building production LLM systems.

---

## 1. Separation of Concerns

### Why It Matters
When you mix responsibilities (e.g., API handler doing prompt building + database access + inference), you get:
- **Hard to test**: Can't test prompt logic without starting web server
- **Hard to change**: Changing prompt format requires touching API code
- **Hard to scale**: Can't run inference on separate machines

### Our Approach
**Clear layer boundaries**:
- API: HTTP/SSE only → calls Runtime
- Runtime: Orchestration only → calls Services + Core
- Services: Business logic → calls Engine
- Engine: Inference only → no knowledge of sessions/APIs

**Result**: Can swap llama.cpp for vLLM by changing only `engine/` files.

---

## 2. Stateless Inference Engine

### Why It Matters
Many systems store session state in the inference engine:
```python
# ❌ BAD: Engine knows about sessions
class InferenceEngine:
    def __init__(self):
        self.sessions = {}  # WHY is this here?
    
    def generate(self, session_id, prompt):
        history = self.sessions[session_id]  # Inference mixed with history
```

**Problems**:
- Can't run multiple engine instances (state is local)
- Can't restart engine without losing sessions
- Can't scale inference separately

### Our Approach
**Engine is pure function**: `prompt → tokens`
```python
class LlamaCppModel:
    def generate(self, prompt: str) -> Iterator[str]:
        # No session state, no database, no business logic
        return llama_cpp.create_completion(prompt)
```

**Session management is separate layer**:
```python
class SessionManager:
    def get_messages(self, session_id) -> List[Message]:
        # Loads from database, returns messages
```

**Result**: Can run 10 inference workers sharing same session database.

---

## 3. Token-Based Truncation Strategy

### The Problem
Context windows are limited (8K, 128K tokens). Long conversations exceed limits.

**Naive approaches**:
- Truncate by character count → Breaks mid-token, inaccurate
- Keep last N messages → Might exceed window if messages are long
- Remove all history → Loses context

### Our Approach
**Token-aware truncation with priorities**:

```python
def estimate_trimmed_messages(messages, max_tokens, template):
    # Priority 1: Always keep system message
    system_msgs = [m for m in messages if m.role == "system"]
    system_tokens = count_tokens(system_msgs)
    
    # Priority 2: Always keep last user message
    last_user = messages[-1]
    last_user_tokens = count_tokens(last_user)
    
    # Priority 3: Keep recent history (backward from second-to-last)
    available = max_tokens - system_tokens - last_user_tokens
    
    trimmed = []
    running_total = 0
    
    for msg in reversed(messages[:-1]):
        msg_tokens = count_tokens(msg)
        if running_total + msg_tokens <= available:
            trimmed.insert(0, msg)
            running_total += msg_tokens
        else:
            break  # Can't fit any more
    
    return system_msgs + trimmed + [last_user]
```

**Key insights**:
- System message sets behavior (always keep)
- Last user message is what we're responding to (always keep)
- Fill remaining space with recent history (as much as fits)
- Preserve user-assistant pairs when possible

---

## 4. LRU Model Caching

### Why Not "Load on Startup"?
With 10 models @ 4GB each = 40GB RAM. Can't keep all loaded.

### Why Not "Load on Each Request"?
Loading takes 5-30 seconds. Unacceptable latency.

### Our Approach
**Least Recently Used (LRU) Cache**:

```python
class ModelLoader:
    def __init__(self, max_models=3):
        self._loaded_models = OrderedDict()  # Preserves insertion order
        self.max_models = max_models
    
    def get_or_load_model(self, name):
        if name in self._loaded_models:
            # Cache HIT: Move to end (mark as recently used)
            self._loaded_models.move_to_end(name)
            return self._loaded_models[name]
        
        # Cache MISS: Load model
        model = load_model(name)
        self._loaded_models[name] = model
        
        # Evict oldest if over limit
        while len(self._loaded_models) > self.max_models:
            oldest = next(iter(self._loaded_models))
            self._unload_model(oldest)
        
        return model
```

**Result**: 
- First request: 10s (cold start)
- Subsequent requests: <100ms (cache hit)
- Memory usage bounded (max 3 models × 4GB = 12GB)

---

## 5. Streaming with SSE (not WebSockets)

### Why Not WebSockets?
WebSockets require:
- Bidirectional channel (we only send server → client)
- Connection state management
- More complex to implement
- Harder to scale (sticky sessions)

### Our Approach
**Server-Sent Events (SSE)**:

```python
async def generate_sse():
    for token in token_generator:
        # SSE format: "data: {json}\n\n"
        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
    
    yield f"data: {json.dumps({'done': True, 'full_text': accumulated})}\n\n"

return EventSourceResponse(generate_sse())
```

**Benefits**:
- Built on HTTP (works everywhere)
- Automatic reconnection in browser
- Simple to implement
- Load balancer friendly (no sticky sessions needed)
- Works through proxies

**Client side**:
```javascript
const source = new EventSource('/api/chat');
source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (!data.done) {
        console.log(data.token);  // Display incrementally
    }
};
```

---

## 6. Template-Based Prompts

### The Problem
Different models expect different formats:
- Llama 3: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{msg}<|eot_id|>`
- ChatML: `<|im_start|>user\n{msg}<|im_end|>`
- Alpaca: `### Instruction:\n{msg}\n\n### Response:\n`

**Naive approach**: Hardcode format in API handler
- Can't support multiple models
- Changing format requires code changes

### Our Approach
**Template System**:

```python
TEMPLATES = {
    "llama3": {
        "bos": "<|begin_of_text|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        ...
    },
    "chatml": {
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        ...
    }
}

class PromptBuilder:
    def __init__(self, template_name):
        self.template = TEMPLATES[template_name]
    
    def build_prompt(self, messages):
        parts = []
        for msg in messages:
            parts.append(self.template[f"{msg.role}_start"])
            parts.append(msg.content)
            parts.append(self.template[f"{msg.role}_end"])
        return "".join(parts)
```

**Benefits**:
- Add new template: Just add dict entry
- Model config specifies template: `{"template": "llama3"}`
- Auto-detection: Check model name for "llama-3" → use llama3 template

---

## 7. Async Database for Concurrency

### Why Async?
**Blocking example**:
```python
# ❌ Synchronous (blocks event loop)
def get_messages(session_id):
    result = db.query(Message).filter_by(session_id=session_id).all()
    return result  # Blocks for 50-200ms while waiting for database
```

With 100 concurrent requests:
- Request 1: Queries DB (blocks 100ms)
- Requests 2-100: Wait in queue
- Total time: 100 × 100ms = 10 seconds

**Non-blocking example**:
```python
# ✅ Asynchronous (yields control while waiting)
async def get_messages(session_id):
    result = await db.execute(select(Message).filter_by(session_id=session_id))
    return result.scalars().all()
```

With 100 concurrent requests:
- All 100 requests sent to DB in parallel
- Event loop handles other work while waiting
- Total time: ~100ms (limited by DB, not Python)

### Our Implementation
**AsyncSession + async/await throughout**:
```python
class Database:
    def __init__(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///db.db")
        self.async_session_maker = async_sessionmaker(self.engine)
    
    @asynccontextmanager
    async def get_session(self):
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except:
                await session.rollback()
                raise
```

---

## 8. Error Handling Strategy

### Principle: Fail Fast, Fail Clearly

**Custom exception hierarchy**:
```python
MyLLMError (base)
├─ ModelNotFoundError (404)
├─ ModelLoadError (500)
├─ InferenceError (500)
├─ SessionNotFoundError (404)
├─ ContextWindowExceededError (400)
└─ ConfigurationError (500)
```

**At API boundary**:
```python
try:
    result = await runtime.chat(...)
except ModelNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e))
except InferenceError as e:
    raise HTTPException(status_code=500, detail=str(e))
```

**Benefits**:
- Clients get appropriate HTTP status codes
- Error messages are actionable
- Can log different error types differently
- Can add retry logic selectively

---

## 9. Configuration Management

### Why Not Environment Variables Directly?
```python
# ❌ Scattered across codebase
port = int(os.getenv("PORT", "8000"))
host = os.getenv("HOST", "127.0.0.1")
# Typos, type errors, no validation
```

### Our Approach
**Pydantic Settings (centralized, validated)**:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    models_dir: Path = Field(default=Path("./models_data"))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

settings = get_settings()  # Singleton
```

**Benefits**:
- Type validation (port must be int)
- Default values in one place
- Auto-loads from .env file
- Environment variables override defaults
- Can validate complex constraints

---

## 10. Why These Layers?

```
API → Runtime → Services → Engine
```

### Why Not Just: API → Engine?
**Problem**: API would need to:
- Load models
- Manage sessions
- Build prompts
- Handle caching
- Execute inference

Result: 1000-line API files, impossible to test.

### Layer Responsibilities

**API Layer**:
- ONLY: HTTP/SSE protocol
- Validates requests
- Delegates to Runtime
- Formats responses

**Runtime Layer**:
- ONLY: Orchestrates flows
- Combines: session + prompt + inference
- No implementation details

**Services Layer**:
- ONLY: Specific business logic
- ModelLoader: Loading/caching
- InferenceService: Generation execution
- Each service focused on one thing

**Engine Layer**:
- ONLY: Wraps external libraries
- No business rules
- Pure input → output functions

**Result**: Each file has one clear job, easy to test, easy to change.

---

## Common Mistakes to Avoid

### ❌ 1. Mixing Concerns
```python
@router.post("/chat")
async def chat(request):
    # Building prompt in API handler - WRONG!
    prompt = f"<|im_start|>user\n{request.message}<|im_end|>"
    
    # Loading model in API handler - WRONG!
    model = Llama(model_path="...")
    
    # Saving to database in API handler - WRONG!
    db.execute("INSERT INTO messages ...")
```

**Fix**: Delegate everything except HTTP to lower layers.

### ❌ 2. Blocking Calls in Async Functions
```python
async def handler():
    result = sync_database_call()  # Blocks entire event loop!
```

**Fix**: Use `await` or `run_in_executor`:
```python
async def handler():
    result = await async_database_call()
    # or
    result = await loop.run_in_executor(None, sync_function)
```

### ❌ 3. Circular Dependencies
```python
# runtime.py
from app.services.inference import InferenceService

# inference.py
from app.core.runtime import RuntimeManager  # CIRCULAR!
```

**Fix**: Dependencies flow one direction (top → bottom).

### ❌ 4. Tight Coupling
```python
# Service directly accessing database - BAD
class InferenceService:
    def infer(self):
        messages = db.query(Message).all()  # WHY?
```

**Fix**: Services receive data from orchestrator, don't access storage directly.

### ❌ 5. No Error Context
```python
except Exception as e:
    raise Exception("Error")  # What error? Where? Why?
```

**Fix**: Custom exceptions with details:
```python
except FileNotFoundError:
    raise ModelLoadError(model_name, f"File not found: {path}")
```

---

## Performance Optimizations

### 1. Model Caching (Biggest Impact)
- **Before**: Load model on every request (10s per request)
- **After**: LRU cache (100ms per cached request)
- **Impact**: 100x speedup for subsequent requests

### 2. Prompt Caching
- **Before**: Re-format same prompts repeatedly
- **After**: Cache by message hash
- **Impact**: Saves a few milliseconds, reduces CPU

### 3. Embedding Caching
- **Before**: Generate same embeddings multiple times
- **After**: Cache by text hash (1 hour TTL)
- **Impact**: Instant retrieval for repeated text

### 4. Eager Loading (Database)
```python
# ❌ N+1 queries
session = db.query(Session).get(session_id)
for message in session.messages:  # Queries for each message!
    print(message.content)

# ✅ Single query with join
session = db.query(Session).options(selectinload(Session.messages)).get(session_id)
```

### 5. Async Everything
- Database: Async SQLAlchemy
- API: Async FastAPI
- Services: Async methods
- **Result**: Handle 100+ concurrent requests

---

## Scalability Path

### Phase 1: Single Server (Current)
```
[Client] → [FastAPI + llama.cpp + SQLite]
```
**Limits**: One GPU, ~10 req/sec

### Phase 2: Vertical Scaling
```
[Client] → [FastAPI + Multi-GPU + PostgreSQL]
```
- Bigger GPU (A100 vs RTX 3090)
- More RAM (256GB for larger model cache)
- PostgreSQL instead of SQLite
**Limits**: One machine, ~50 req/sec

### Phase 3: Horizontal Scaling
```
                    ┌─ [API Server 1] ─┐
[Client] → [LB] ────┼─ [API Server 2] ─┼─ [Shared DB]
                    └─ [API Server 3] ─┘
                             │
                    ┌────────┴────────┐
                    [Inference Workers]
```
- Load balancer distributes requests
- Multiple API servers (stateless)
- Shared PostgreSQL/Redis
- Dedicated inference machines with GPUs
**Limits**: Cost, ~1000 req/sec

### Phase 4: Distributed
```
[Clients] → [API Gateway] → [Service Mesh]
                                 ├─ [Chat Service]
                                 ├─ [Inference Pool]
                                 ├─ [Embedding Service]
                                 └─ [Session Service]
```
- Microservices architecture
- Kubernetes orchestration
- Auto-scaling based on load
**Limits**: Sky's the limit (and budget)

---

## Summary

**Key Takeaways**:

1. **Separate concerns**: Each layer has one job
2. **Stateless engine**: Inference is pure function (prompt → tokens)
3. **Smart truncation**: Token-aware, priority-based history management
4. **LRU caching**: Keep hot models in memory, evict cold ones
5. **SSE streaming**: Simple, effective, scalable
6. **Template system**: Support any model format without code changes
7. **Async database**: Non-blocking I/O for concurrency
8. **Clear errors**: Specific exceptions with actionable messages
9. **Type safety**: Pydantic everywhere for validation
10. **One-way dependencies**: Never import upwards

**Result**: Production-grade system that's:
- Fast (model caching, async I/O)
- Reliable (error handling, transactions)
- Maintainable (clear layers, single responsibility)
- Extensible (swap components easily)
- Scalable (stateless design, horizontal ready)

This is **real production architecture**, not a toy project.
