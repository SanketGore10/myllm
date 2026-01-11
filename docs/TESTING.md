# MyLLM Testing Guide

## Installation Status

Currently installing dependencies. The `llama-cpp-python` package is building from source, which takes 5-10 minutes because it compiles C++ code for the llama.cpp inference engine.

## What We Can Test

### Phase 1: Without Models (Infrastructure Testing)
These tests verify the system works without requiring actual GGUF models:

1. **CLI Installation**
   ```bash
   # Verify myllm command is available
   myllm --help
   ```

2. **List Models (Empty)**
   ```bash
   # Should show "No models found" message
   myllm list
   ```

3. **API Server Startup**
   ```bash
   # Start the server (should work even without models)
   myllm serve
   # Visit http://localhost:8000
   # Visit http://localhost:8000/docs (API documentation)
   ```

4. **Health Check**
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy"}
   ```

5. **Python Import Test**
   ```python
   # Test all modules import correctly
   from app.core.config import get_settings
   from app.models.registry import get_registry
   from app.storage.database import get_db
   # etc.
   ```

### Phase 2: With a Small Test Model
To fully test the system, you need to download a GGUF model:

**Recommended Small Model for Testing** (~3-4GB):
- **TinyLlama 1.1B Q4_K_M** - Fast, small, good for testing
- **Phi-2 Q4_K_M** - Also small and fast

**Download Example** (TinyLlama):
```bash
# Create model directory
mkdir -p models_data/tinyllama-1.1b
cd models_data/tinyllama-1.1b

# Download from Hugging Face
# Visit: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Download: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Rename to model.gguf
mv tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf model.gguf

# Create config.json
cat > config.json << 'EOF'
{
  "name": "tinyllama-1.1b",
  "family": "llama",
  "quantization": "Q4_K_M",
  "context_size": 2048,
  "template": "chatml",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9
  }
}
EOF
```

### Phase 3: Full Functionality Testing

Once you have a model:

1. **List Models**
   ```bash
   myllm list
   # Should show: tinyllama-1.1b
   ```

2. **Show Model Details**
   ```bash
   myllm show tinyllama-1.1b
   # Should display: size, quantization, context size, etc.
   ```

3. **Interactive Chat (CLI)**
   ```bash
   myllm run tinyllama-1.1b
   # Try: "Hello, how are you?"
   # Try: "What is 2+2?"
   # Try: "/help" to see commands
   # Try: "/exit" to quit
   ```

4. **API Chat (Non-Streaming)**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": false
     }'
   ```

5. **API Chat (Streaming SSE)**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "messages": [{"role": "user", "content": "Tell me a joke"}],
       "stream": true
     }'
   ```

6. **API Generate**
   ```bash
   curl -X POST http://localhost:8000/api/generate \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "prompt": "Once upon a time",
       "stream": false,
       "options": {"max_tokens": 50}
     }'
   ```

7. **API Embeddings**
   ```bash
   curl -X POST http://localhost:8000/api/embeddings \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "input": "Hello world"
     }'
   ```

8. **Session Continuity Test**
   ```bash
   # First message
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "messages": [{"role": "user", "content": "My name is Alice"}],
       "stream": false
     }'
   # Note the session_id in response
   
   # Second message (should remember)
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "model": "tinyllama-1.1b",
       "messages": [{"role": "user", "content": "What is my name?"}],
       "session_id": "<session_id_from_above>",
       "stream": false
     }'
   ```

## Expected Results

### ✅ Success Indicators

1. **Installation**: `myllm --help` shows commands
2. **Server Startup**: Logs show "Application startup complete"
3. **API Docs**: http://localhost:8000/docs shows Swagger UI
4. **Health Check**: Returns `{"status": "healthy"}`
5. **Model Loading**: First inference takes 5-30 seconds (loading model)
6. **Subsequent Requests**: <1 second (model cached)
7. **Streaming**: Tokens appear incrementally in real-time
8. **Session**: Second message remembers context from first

### ❌ Common Issues

1. **Import Errors**
   - Solution: Ensure all dependencies installed: `pip install -e .`

2. **Model Not Found**
   - Solution: Check `models_data/<name>/model.gguf` exists
   - Solution: Verify `config.json` has correct name

3. **Out of Memory**
   - Solution: Use smaller model or Q4_0 quantization
   - Solution: Reduce `max_loaded_models` in .env

4. **Slow Inference**
   - Check: CPU-only mode (expected if no GPU)
   - Solution: Install llama-cpp-python with GPU support

5. **Database Locked**
   - Solution: Only one server instance at a time
   - Solution: Delete `myllm.db` to reset

## Performance Benchmarks

With **TinyLlama 1.1B Q4_K_M** on typical hardware:

| Metric | CPU (i7) | GPU (RTX 3060) |
|--------|----------|----------------|
| Model Load | ~5 sec | ~3 sec |
| First Token | ~500ms | ~100ms |
| Tokens/sec | ~5-10 | ~30-50 |
| Memory | ~2GB RAM | ~2GB VRAM |

## Testing Checklist

- [ ] Installation completes successfully
- [ ] `myllm --help` works
- [ ] `myllm list` shows no models (before download)
- [ ] `myllm serve` starts without errors
- [ ] Health endpoint returns 200
- [ ] API docs accessible at /docs
- [ ] Download and setup a test model
- [ ] `myllm list` shows the model
- [ ] `myllm show <model>` displays details
- [ ] `myllm run <model>` starts interactive chat
- [ ] Interactive chat responds to messages
- [ ] CLI special commands work (/help, /clear, /exit)
- [ ] API chat (non-streaming) returns response
- [ ] API chat (streaming) returns SSE events
- [ ] API generate works
- [ ] API embeddings returns vector
- [ ] Session continuity works (remembers context)
- [ ] Multiple concurrent requests work
- [ ] Model caching works (second request faster)

## Next Steps After Basic Testing

1. **Load Testing**
   - Use Apache Bench or wrk to test concurrent requests
   - Example: `ab -n 100 -c 10 http://localhost:8000/health`

2. **GPU Testing** (if available)
   ```bash
   # Reinstall with CUDA support
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
   # Verify GPU usage during inference
   nvidia-smi  # Watch GPU memory/utilization
   ```

3. **Production Deployment**
   - Set up reverse proxy (nginx)
   - Configure systemd service
   - Enable HTTPS
   - Set up monitoring (Prometheus)

4. **Try Larger Models**
   - Llama 3 8B (~4-5GB)
   - Mistral 7B (~4GB)
   - Check context window limits

## Troubleshooting Commands

```bash
# Check Python version
python --version  # Should be >=3.10

# Check installed packages
pip list | grep llama

# Check if myllm is installed
which myllm  # or: where myllm (Windows)

# Check logs
# Logs are printed to console when running

# Reset database
rm myllm.db

# Check model files
ls -lh models_data/*/model.gguf

# Test database directly
python -c "from app.storage.database import get_db; import asyncio; asyncio.run(get_db().init_db())"
```

## Notes

- **First Run**: Slower due to model loading (5-30 seconds)
- **Cached Runs**: Much faster (<1 second)
- **Memory**: Keep ~2x model size available (4GB model needs ~8GB RAM)
- **GPU**: Optional but dramatically faster (10-50x speedup)
- **Context**: Longer conversations auto-truncate to fit window
