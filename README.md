# MyLLM - Production-Grade Local LLM Runtime

A custom-built local LLM runtime system, designed for learning, extensibility, and production use. Run large language models locally with streaming chat, embeddings, and a clean REST API + CLI interface.

## Features

- 🚀 **Local Inference**: Run LLMs locally using llama.cpp with GGUF models
- 💬 **Chat API**: Streaming chat with conversation history management
- 🔄 **Generation**: Single-shot text generation
- 🎯 **Embeddings**: Generate text embeddings
- ⚡ **GPU Acceleration**: Automatic GPU detection and layer offloading
- 📝 **Session Management**: Persistent conversation history with context window truncation
- 🌊 **Streaming**: Server-Sent Events (SSE) for real-time token streaming
- 🛠️ **CLI**: Interactive chat and server management
- 🎨 **Clean Architecture**: Separated concerns, extensible, maintainable

## Architecture

```
Client (CLI/API) → API Layer (FastAPI) → Service Layer → Engine Layer (llama.cpp)
                                      ↓
                              Storage Layer (SQLite)
```

**Key Components**:
- **API Layer**: REST endpoints for chat, generate, models, embeddings
- **Core**: Runtime orchestration, session management, prompt building
- **Engine**: llama.cpp wrapper with streaming support
- **Services**: Model loading, inference execution, embeddings
- **Storage**: SQLite for session persistence

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/myllm.git
cd myllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For GPU support (CUDA example)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For Mac (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Setup

1. **Configure environment**:
```bash
cp .env.example .env
# Edit .env to set MODELS_DIR, PORT, etc.
```

2. **Download a model** (manually for now):
```bash
mkdir -p models_data/llama-3-8b
cd models_data/llama-3-8b

# Download GGUF model (example)
wget https://huggingface.co/.../model.gguf

# Create config.json
cat > config.json << EOF
{
  "name": "llama-3-8b",
  "family": "llama",
  "quantization": "Q4_K_M",
  "context_size": 8192,
  "template": "llama3",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9
  }
}
EOF
```

### Usage

#### Start API Server

```bash
myllm serve
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Interactive Chat (CLI)

```bash
myllm run llama-3-8b
```

```
You: Hello!
Assistant: Hello! How can I help you today?

You: What's the capital of France?
Assistant: The capital of France is Paris.

You: /exit
Goodbye!
```

#### API Usage

**Chat (Streaming)**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

Response (SSE stream):
```
data: {"token":"Hello","done":false}

data: {"token":"!","done":false}

data: {"done":true,"session_id":"123e4567-e89b-12d3-a456-426614174000","full_text":"Hello! How can I help you today?"}
```

**Chat (Non-Streaming)**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

**Generate**:
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "prompt": "Once upon a time",
    "options": {
      "max_tokens": 100,
      "temperature": 0.8
    }
  }'
```

**List Models**:
```bash
curl http://localhost:8000/api/models
```

**Embeddings**:
```bash
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "input": "Hello world"
  }'
```

## Project Structure

```
myllm/
├── app/
│   ├── main.py                 # FastAPI application factory
│   ├── api/                    # REST API endpoints
│   │   ├── chat.py            # Chat endpoint with streaming
│   │   ├── generate.py        # Single-shot generation
│   │   ├── embeddings.py      # Embeddings endpoint
│   │   └── models.py          # Model management
│   ├── core/                   # Core business logic
│   │   ├── config.py          # Configuration management
│   │   ├── runtime.py         # Runtime orchestrator
│   │   ├── session.py         # Session & history management
│   │   └── prompt.py          # Prompt template builder
│   ├── engine/                 # Inference engine layer
│   │   ├── llama_cpp.py       # llama.cpp wrapper
│   │   ├── tokenizer.py       # Token counting
│   │   └── streaming.py       # SSE streaming handler
│   ├── models/                 # Model management
│   │   ├── registry.py        # Model discovery & registry
│   │   └── schemas.py         # Pydantic data models
│   ├── services/               # Service layer
│   │   ├── model_loader.py    # Model loading with caching
│   │   ├── inference.py       # Inference orchestration
│   │   └── embeddings.py      # Embedding generation
│   ├── storage/                # Data persistence
│   │   ├── database.py        # SQLite ORM models
│   │   └── cache.py           # In-memory caching
│   └── utils/                  # Utilities
│       ├── hardware.py        # Hardware detection
│       ├── logging.py         # Logging configuration
│       └── errors.py          # Custom exceptions
├── cli/                        # CLI interface
│   ├── main.py                # CLI entry point (Typer)
│   └── commands/              # CLI commands
│       ├── pull.py            # Download models
│       ├── run.py             # Interactive chat
│       └── serve.py           # Start API server
├── models_data/                # Model storage
│   └── <model_name>/
│       ├── model.gguf
│       └── config.json
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
└── docs/                       # Documentation
```

## Configuration

Edit `.env`:

```bash
# Server
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# Paths
MODELS_DIR=./models_data
DB_PATH=./myllm.db

# Inference
DEFAULT_CONTEXT_SIZE=4096
DEFAULT_N_GPU_LAYERS=-1        # -1 = use all GPU layers
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=512

# Performance
MAX_LOADED_MODELS=3            # LRU cache size
```

## Model Configuration

Each model needs a `config.json`:

```json
{
  "name": "llama-3-8b",
  "family": "llama",
  "quantization": "Q4_K_M",
  "context_size": 8192,
  "template": "llama3",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.1
  }
}
```

**Supported Templates**:
- `llama3` - Llama 3 format
- `chatml` - ChatML format (default)
- `alpaca` - Alpaca format
- `vicuna` - Vicuna format

## Development

### Setup Dev Environment

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest --cov=app tests/
```

### Code Quality

```bash
# Format
black app/ cli/ tests/

# Lint
ruff check app/ cli/ tests/

# Type check
mypy app/ cli/
```

## API Reference

### Chat Endpoint

**POST /api/chat**

Request:
```typescript
{
  model: string;
  messages: Array<{role: "system" | "user" | "assistant", content: string}>;
  session_id?: string;    // Optional: resume conversation
  stream?: boolean;       // Default: true
  options?: {
    temperature?: number;
    top_p?: number;
    max_tokens?: number;
    stop?: string[];
  };
}
```

Response (streaming):
```
data: {"token": "...", "done": false}
data: {"done": true, "session_id": "...", "full_text": "..."}
```

Response (non-streaming):
```json
{
  "message": {"role": "assistant", "content": "..."},
  "session_id": "...",
  "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
}
```

### Generate Endpoint

**POST /api/generate**

Request:
```json
{
  "model": "llama-3-8b",
  "prompt": "Once upon a time",
  "stream": false,
  "options": {
    "max_tokens": 100,
    "temperature": 0.8
  }
}
```

### Models Endpoint

**GET /api/models**

Returns list of available models.

**GET /api/models/{model_name}**

Returns detailed model information.

## Performance Tips

1. **GPU Offloading**: Set `n_gpu_layers=-1` to offload all layers to GPU
2. **Quantization**: Use Q4_K_M or Q5_K_M for good balance of speed/quality
3. **Context Size**: Reduce `context_size` if running out of VRAM
4. **Batch Size**: Adjust `n_batch` in llama.cpp for your hardware
5. **KV Cache**: Enable by default for faster multi-turn conversations

## Troubleshooting

### Model Not Loading

- Check model path in `MODELS_DIR`
- Verify `config.json` exists
- Check llama.cpp Python bindings installed correctly

### GPU Not Detected

- Reinstall llama-cpp-python with correct CMAKE flags
- Check CUDA/Metal/ROCm drivers installed
- Verify `nvidia-smi` or equivalent shows GPU

### Out of Memory

- Reduce `n_gpu_layers`
- Use more aggressive quantization (Q4_0)
- Reduce `context_size`
- Limit `MAX_LOADED_MODELS`

## Roadmap

- [ ] Automatic model downloads from Hugging Face
- [ ] Vision model support (LLaVA)
- [ ] Function calling / tools
- [ ] Multi-model routing
- [ ] RAG (retrieval-augmented generation)
- [ ] Model fine-tuning integration
- [ ] Distributed inference
- [ ] Web UI dashboard

## License

MIT

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Acknowledgments

- **llama.cpp**: Incredible inference engine
- **FastAPI**: Modern Python web framework
- **Ollama**: Inspiration for this project
