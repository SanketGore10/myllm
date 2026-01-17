Assistant: hi there! how are you doing today?
────────────────────────────────────────────────────

You: /clear
History cleared. Starting new session.

You: /exit
Goodbye!
```

**Interactive Commands:**
- `/exit` - Exit the chat session
- `/clear` - Clear conversation history, start fresh
- `/help` - Show command help

**Keyboard Shortcuts:**
- `Ctrl+C` (once) - Interrupt current generation, stay in chat
- `Ctrl+C` (twice) - Exit chat immediately
- `Ctrl+D` - Exit chat (EOF)

---

### 6. `myllm serve` - Start API Server

Start the FastAPI web server.

**Basic Usage:**
```bash
$ myllm serve
```

**With Custom Host/Port:**
```bash
$ myllm serve --host 0.0.0.0 --port 8080
```

**With Auto-Reload (Development):**
```bash
$ myllm serve --reload
```

**Options:**
- `--host` - Server host address (default: 127.0.0.1)
- `--port` - Server port (default: 8000)
- `--reload` - Auto-reload on code changes

**Access Points:**
- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

**To Stop the Server:**
- Press `Ctrl+C` in the terminal
- Or send kill signal: `kill <process-id>`

---

## API Usage Examples

### Start Server First
```bash
$ myllm serve
```

### 1. Chat (Non-Streaming)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### 2. Chat (Streaming with SSE)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "messages": [
      {"role": "user", "content": "Tell me a joke"}
    ],
    "stream": true
  }'
```

**Response (SSE Stream):**
```
data: {"token":"Why","done":false}

data: {"token":" did","done":false}

data: {"token":" the","done":false}

data: {"done":true,"session_id":"uuid","full_text":"Why did the..."}
```

### 3. Continue Conversation (Session)

```bash
# First message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "messages": [
      {"role": "user", "content": "My name is Alice"}
    ],
    "stream": false
  }'
# Save the session_id from response

# Second message (remembers context)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "messages": [
      {"role": "user", "content": "What is my name?"}
    ],
    "session_id": "your-session-id-here",
    "stream": false
  }'
```

### 4. Generate Text

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "prompt": "Once upon a time",
    "stream": false,
    "options": {
      "max_tokens": 100,
      "temperature": 0.8
    }
  }'
```

### 5. Get Embeddings

```bash
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[model-name]",
    "input": "Hello world"
  }'
```

**Response:**
```json
{
  "embeddings": [0.123, -0.456, 0.789, ...],
  "model": "[model-name]"
}
```

### 6. List Models

```bash
curl http://localhost:8000/api/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "[model-name]",
      "family": "llama",
      "quantization": "Q4_K_M",
      "context_size": 2048,
      "size_bytes": 668123136,
      "status": "available"
    }
  ]
}
```

### 7. Get Model Details

```bash
curl http://localhost:8000/api/models/[model-name]
```

---

## Python Client Example

```python
import requests

# Non-streaming chat
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "model": "[model-name]",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False
    }
)

data = response.json()
print(data["message"]["content"])

# Streaming chat
import sseclient

response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "model": "[model-name]",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    data = json.loads(event.data)
    if not data.get("done"):
        print(data["token"], end="", flush=True)
```

---

## JavaScript Client Example

```javascript
// Non-streaming
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: '[model-name]',
    messages: [{role: 'user', content: 'Hello!'}],
    stream: false
  })
});

const data = await response.json();
console.log(data.message.content);

// Streaming with EventSource
const eventSource = new EventSource('http://localhost:8000/api/chat');
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (!data.done) {
    console.log(data.token);
  } else {
    console.log('\n\nSession ID:', data.session_id);
    eventSource.close();
  }
};
```

---

## Exit & Stop Commands Summary

### Exiting Interactive Chat
- Type: `/exit`
- Keyboard: `Ctrl+C` (twice) or `Ctrl+D`

### Stopping API Server
- Keyboard: `Ctrl+C` in terminal
- Command: `kill <process-id>`
- Signal: `pkill -f "myllm serve"`

### Interrupting Generation
- Keyboard: `Ctrl+C` (once) - Stops current response, stays in chat

---

## Common Options

### Temperature Setting
Controls randomness in generation:
- `0.0-0.3`: Very focused, deterministic
- `0.4-0.7`: Balanced (default: 0.7)
- `0.8-1.2`: Creative, varied
- `1.3-2.0`: Very random, experimental

**CLI:**
```bash
myllm run [model-name] --temperature 0.5
```

**API:**
```json
{
  "options": {
    "temperature": 0.5
  }
}
```

---

## Troubleshooting

### Model Not Found
```bash
$ myllm list                    # Check installed models
$ myllm pull --list             # See available models
$ myllm pull [model-name]     # Download missing model
```

### Server Already Running
```bash
# Find process
ps aux | grep "myllm serve"

# Kill it
kill <process-id>

# Or use killall
killall -9 uvicorn
```

### Port Already In Use
```bash
# Use different port
myllm serve --port 8080
```

---

## Quick Start Checklist

- [ ] Install: `pip install -e .`
- [ ] Test: `myllm --help`
- [ ] List models: `myllm pull --list`
- [ ] Download: `myllm pull [model-name]`
- [ ] Verify: `myllm list`
- [ ] Chat: `myllm run [model-name]`
- [ ] API: `myllm serve` (in another terminal)
- [ ] Test API: Visit `http://localhost:8000/docs`

---

## Advanced Usage

### Custom Model Directory
```bash
myllm --models-dir /path/to/models list
```

### Verbose Logging
```bash
myllm --verbose run [model-name]
myllm -v serve
```

---

## Documentation Links

- **Full README**: `README.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **API Reference**: `docs/API.md`
- **Testing Guide**: `docs/TESTING.md`
- **Design Insights**: `docs/DESIGN_INSIGHTS.md`

---

## Support

- GitHub Issues: https://github.com/SanketGore10/myllm/issues
- API Docs: http://localhost:8000/docs (when server running)

---

**Last Updated**: 2026-01-11
