# MyLLM Architecture Documentation

## System Overview

MyLLM is a production-grade local LLM runtime system built with clean architecture principles. It enables running large language models locally using llama.cpp with support for chat, generation, and embeddings.

## Architecture Layers

### 1. Presentation Layer
- **API (FastAPI)**: REST endpoints with SSE streaming
- **CLI (Typer)**: Interactive terminal interface

### 2. Service Layer
- **Model Loader**: LRU caching of loaded models
- **Inference Service**: Coordinates generation execution
- **Embeddings Service**: Manages embedding generation

### 3. Core Business Logic
- **Runtime Manager**: High-level orchestrator
- **Session Manager**: Conversation history with truncation
- **Prompt Builder**: Template-based prompt formatting

### 4. Engine Layer
- **llama.cpp Wrapper**: Direct integration with llama-cpp-python
- **Tokenizer**: Token counting and truncation
- **Streaming**: SSE event formatting

### 5. Models Layer
- **Registry**: Model discovery and metadata
- **Schemas**: Pydantic data validation

### 6. Storage Layer
- **Database**: SQLite with async SQLAlchemy
- **Cache**: In-memory LRU cache

### 7. Utilities
- **Config**: Environment-based settings
- **Logging**: Structured logging with request tracking
- **Hardware**: GPU/CPU detection
- **Errors**: Custom exception hierarchy

## Request Flow: Chat with Streaming

```
1. HTTP Request → POST /api/chat
   ↓
2. API Handler (chat.py)
   - Validates request schema
   - Extracts parameters
   ↓
3. Runtime Manager (runtime.py)
   - Creates/loads session
   - Gets existing history from database
   - Combines history + new message
   ↓
4. Session Manager (session.py)
   - Loads messages from database
   - Counts tokens per message
   - Truncates to fit context window
   ↓
5. Prompt Builder (prompt.py)
   - Applies model-specific template
   - Formats system/user/assistant messages
   - Generates final prompt string
   ↓
6. Inference Service (inference.py)
   - Merges user options with defaults
   - Calls model loader
   ↓
7. Model Loader (model_loader.py)
   - Checks LRU cache
   - Loads model if not cached
   - Evicts oldest if over limit
   ↓
8. llama.cpp Engine (llama_cpp.py)
   - Executes llama_cpp_python.create_completion()
   - Yields tokens via generator
   ↓
9. Streaming Handler (streaming.py)
   - Converts tokens to SSE format
   - Accumulates full response
   ↓
10. API Handler (continued)
    - Streams SSE events to client
    - Saves conversation turn after completion
    ↓
11. Client receives SSE stream
    data: {"token": "Hello", "done": false}
    data: {"token": " world", "done": false}
    data: {"done": true, "session_id": "...", "full_text": "Hello world"}
```

## Key Design Decisions

### 1. Stateless Inference Engine
**Why**: Simplifies concurrency, allows horizontal scaling
- Engine only executes generation
- No session state in engine
- Session management handled separately

### 2. LRU Model Caching
**Why**: Balance memory usage with performance
- Keep frequently used models in memory
- Auto-evict least recently used
- Configurable cache size

### 3. Token-Based Truncation
**Why**: Ensure prompts fit in context window
- Count tokens per message
- Remove oldest messages first
- Always preserve system + last user message

### 4. Template-Based Prompts
**Why**: Support diverse model formats
- ChatML, Llama 3, Alpaca, Vicuna, Mistral
- Auto-detection from model name
- Cached compiled templates

### 5. Async Database Layer
**Why**: Non-blocking I/O for better concurrency
- SQLAlchemy async engine
- Proper connection pooling
- Transaction management

### 6. SSE for Streaming
**Why**: Standard, simple, works everywhere
- Server-Sent Events (EventSource)
- No WebSocket complexity
- Compatible with HTTP/1.1

## Module Responsibilities

### app/api/
- **chat.py**: ONLY handles HTTP/SSE for chat
- **generate.py**: ONLY handles HTTP/SSE for generation
- **models.py**: ONLY handles model listing/management
- **embeddings.py**: ONLY handles embedding requests
- MUST NOT: Contain business logic, session management, or inference

### app/core/
- **runtime.py**: ONLY orchestrates high-level flows
- **session.py**: ONLY manages conversation history
- **prompt.py**: ONLY formats prompts with templates
- **config.py**: ONLY loads and validates settings
- MUST NOT: Execute inference, make HTTP calls, access database directly

### app/engine/
- **llama_cpp.py**: ONLY wraps llama-cpp-python
- **tokenizer.py**: ONLY counts tokens and truncates
- **streaming.py**: ONLY formats SSE events
- MUST NOT: Know about sessions, APIs, or business rules

### app/services/
- **model_loader.py**: ONLY manages model lifecycle
- **inference.py**: ONLY executes generation
- **embeddings.py**: ONLY generates embeddings
- MUST NOT: Know about HTTP, sessions, or prompts

### app/storage/
- **database.py**: ONLY handles database operations
- **cache.py**: ONLY manages in-memory cache
- MUST NOT: Know about models, inference, or APIs

## Common Pitfalls

### ❌ WRONG: API handler contains business logic
```python
@router.post("/chat")
async def chat(request):
    # DON'T build prompts here
    # DON'T manage sessions here
    # DON'T load models here
```

### ✅ CORRECT: API handler delegates to runtime
```python
@router.post("/chat")
async def chat(request):
    runtime = get_runtime()
    generator, session_id = await runtime.chat(...)
    return stream(generator)
```

### ❌ WRONG: Circular dependencies
```python
# runtime.py imports from session.py
# session.py imports from runtime.py
```

### ✅ CORRECT: Unidirectional dependencies
```
API → Runtime → Services → Engine
      ↓
    Session → Database
      ↓
    Prompt → Cache
```

### ❌ WRONG: Mixing sync and async incorrectly
```python
async def handler():
    result = sync_function()  # Blocks event loop!
```

### ✅ CORRECT: Use async throughout or run_in_executor
```python
async def handler():
    result = await async_function()
    # or
    result = await loop.run_in_executor(None, sync_function)
```

## Performance Considerations

### GPU Offloading
- Set `n_gpu_layers=-1` for full GPU offloading
- Detect available GPU automatically
- Fall back to CPU if no GPU

### Context Window Management
- Reserve tokens for generation (max_tokens + buffer)
- Truncate history to fit remaining budget
- Use efficient token counting heuristics

### Model Loading
- Lazy loading on first use
- LRU eviction prevents memory bloat
- Preload frequently used models

### Caching
- Prompt templates cached (rarely change)
- Embeddings cached by text hash
- Model metadata cached in registry

## Extensibility Points

### 1. Add New Inference Backend
Create `app/engine/vllm.py`:
```python
class VLLMModel:
    def generate(self, prompt): ...
    def embed(self, text): ...
```
Update `model_loader.py` to support backend selection

### 2. Add New Prompt Template
Update `app/core/prompt.py`:
```python
TEMPLATES["custom"] = {
    "system_start": "...",
    "user_start": "...",
    ...
}
```

### 3. Add New API Endpoint
Create `app/api/new_endpoint.py`:
```python
router = APIRouter(prefix="/api", tags=["new"])

@router.post("/new")
async def new_endpoint(request):
    runtime = get_runtime()
    result = await runtime.new_operation(...)
    return result
```
Register in `app/api/__init__.py`

### 4. Add New Storage Backend
Create `app/storage/postgres.py`:
```python
class PostgresDatabase:
    async def create_session(self, model_name): ...
    async def add_message(self, session_id, role, content): ...
```
Update `config.py` to support database selection

## Testing Strategy

### Unit Tests
- Test prompt building with different templates
- Test token counting accuracy
- Test truncation logic
- Mock llama.cpp for reproducibility

### Integration Tests
- Test full chat flow with mock model
- Test database persistence
- Test model loading/unloading
- Test SSE streaming format

### End-to-End Tests
- Start real API server
- Send actual HTTP requests
- Verify streaming responses
- Test CLI commands

### Load Tests
- Concurrent requests (100+)
- Multiple models loaded
- Large context windows
- Long-running sessions

## Production Deployment

### Requirements
- Python 3.10+
- llama-cpp-python with GPU support
- SQLite or PostgreSQL
- Reverse proxy (nginx)
- Process manager (systemd, supervisor)

### Recommended Setup
```bash
# 1. Install with GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# 2. Configure environment
cp .env.example .env
# Edit .env with production settings

# 3. Run with process manager
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 4. Nginx reverse proxy
# Proxy to localhost:8000
# Enable gzip compression
# Set appropriate timeouts for streaming
```

### Monitoring
- Prometheus metrics for request rates, latencies
- Log aggregation (structured JSON logs)
- GPU utilization tracking
- Model cache hit/miss rates

### Scaling
- Horizontal: Multiple API instances behind load balancer
- Vertical: Larger GPU, more RAM for model cache
- Distributed: Shared cache (Redis), separate inference workers
