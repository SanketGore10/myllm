# API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication. Add API keys for production.

---

## Endpoints

### GET /
Server information and status.

**Response**:
```json
{
  "name": "MyLLM",
  "version": "0.1.0",
  "status": "running",
  "models_available": 2
}
```

---

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy"
}
```

---

### GET /api/models
List all available models.

**Response**:
```json
{
  "models": [
    {
      "name": "llama-3-8b",
      "family": "llama",
      "size_mb": 4368,
      "quantization": "Q4_K_M",
      "context_size": 8192,
      "template": "llama3",
      "parameters": {},
      "loaded": true
    }
  ]
}
```

---

### GET /api/models/{model_name}
Get detailed information about a specific model.

**Response**:
```json
{
  "name": "llama-3-8b",
  "family": "llama",
  "size_mb": 4368,
  "quantization": "Q4_K_M",
  "context_size": 8192,
  "template": "llama3",
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9
  },
  "loaded": true
}
```

---

### POST /api/models/{model_name}/load
Preload a model into memory.

**Response**:
```json
{
  "status": "success",
  "message": "Model 'llama-3-8b' loaded"
}
```

---

### POST /api/models/{model_name}/unload
Unload a model from memory.

**Response**:
```json
{
  "status": "success",
  "message": "Model 'llama-3-8b' unloaded"
}
```

---

### POST /api/chat
Chat with a model (with conversation history).

**Request**:
```json
{
  "model": "llama-3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "session_id": "optional-session-id",
  "stream": true,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512
  }
}
```

**Response (streaming)**:
```
data: {"token":"Hello","done":false}

data: {"token":"!","done":false}

data: {"token":" How","done":false}

data: {"done":true,"session_id":"123e4567-e89b-12d3-a456-426614174000","full_text":"Hello! How can I help you?"}
```

**Response (non-streaming)**:
```json
{
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you?"
  },
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

---

### POST /api/generate
Generate text from a prompt (stateless).

**Request**:
```json
{
  "model": "llama-3-8b",
  "prompt": "Once upon a time",
  "stream": false,
  "options": {
    "temperature": 0.8,
    "max_tokens": 100
  }
}
```

**Response (non-streaming)**:
```json
{
  "text": "Once upon a time, in a land far away...",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 10,
    "total_tokens": 14
  }
}
```

---

### POST /api/embeddings
Generate embedding vector for text.

**Request**:
```json
{
  "model": "llama-3-8b",
  "input": "Hello world"
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "llama-3-8b"
}
```

---

## Request/Response Schemas

### Message
```typescript
{
  role: "system" | "user" | "assistant",
  content: string
}
```

### InferenceOptions
```typescript
{
  temperature?: number,      // 0.0 to 2.0
  top_p?: number,           // 0.0 to 1.0
  top_k?: number,           // integer >= 0
  max_tokens?: number,      // integer >= 1
  stop?: string[],          // stop sequences
  repeat_penalty?: number,   // >= 0.0
}
```

---

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message description"
}
```

### Status Codes
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found (model doesn't exist)
- `500` - Internal server error

---

## Examples

### Python
```python
import httpx

# Chat (streaming)
with httpx.stream(
    "POST",
    "http://localhost:8000/api/chat",
    json={
        "model": "llama-3-8b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    }
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            print(line[6:])

# Generate (non-streaming)
response = httpx.post(
    "http://localhost:8000/api/generate",
    json={
        "model": "llama-3-8b",
        "prompt": "Once upon a time",
        "stream": False
    }
)
print(response.json()["text"])
```

### JavaScript
```javascript
// Chat (streaming)
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'llama-3-8b',
    messages: [{role: 'user', content: 'Hello!'}],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  console.log(text);
}
```

### cURL
```bash
# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Generate
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "prompt": "Once upon a time",
    "stream": false
  }'

# Embeddings
curl -X POST http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "input": "Hello world"
  }'
```
