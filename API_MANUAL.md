# MyLLM API Documentation

**Version**: 1.0  
**Base URL**: `http://localhost:8000`

MyLLM is a local LLM runtime API compatible with OpenAI-style endpoints. Run powerful language models locally with full privacy and control.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
   - [POST /api/generate](#post-apigenerate)
   - [POST /api/chat](#post-apichat)
   - [POST /api/embeddings](#post-apiembeddings)
   - [GET /api/models](#get-apimodels)
4. [Code Examples](#code-examples)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/SanketGore10/myllm.git
cd myllm

# Install dependencies
pip install -e .

# Download a model
myllm pull tinyllama-1.1b
```

### 2. Start the Server

```bash
myllm serve
```

Server starts at `http://localhost:8000`  
API docs available at `http://localhost:8000/docs`

### 3. Make Your First Request

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b",
    "prompt": "Hello, world!",
    "stream": false
  }'
```

---

## Authentication

**Currently**: No authentication required (local-only API)

**Production**: If exposing publicly, use a reverse proxy (nginx, Caddy) with API key authentication.

---

## API Endpoints

### POST /api/generate

**Stateless text generation** - Generate text from a prompt without conversation history.

#### Request

```json
{
  "model": "tinyllama-1.1b",
  "prompt": "Once upon a time",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop": ["</s>"]
  }
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | ✅ | Model name (from `myllm list`) |
| `prompt` | string | ✅ | Input text prompt |
| `stream` | boolean | ❌ | Enable streaming (default: `true`) |
| `options` | object | ❌ | Generation parameters |

#### Options

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `temperature` | float | 0.7 | 0.0-2.0 | Sampling temperature (higher = more random) |
| `top_p` | float | 0.9 | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | int | 40 | 0+ | Top-K sampling |
| `max_tokens` | int | 512 | 1+ | Maximum tokens to generate |
| `stop` | string[] | `[]` | - | Additional stop sequences |
| `repeat_penalty` | float | 1.1 | 0.0+ | Repetition penalty |

#### Response (Non-Streaming)

```json
{
  "text": "Once upon a time, in a magical forest...",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

#### Response (Streaming)

Server-Sent Events (SSE) stream:

```
data: {"token": "Once", "done": false}
data: {"token": " upon", "done": false}
data: {"token": " a", "done": false}
...
data: {"done": true, "full_text": "Once upon a time...", "token_count": 50}
```

---

### POST /api/chat

**Conversational chat** - Multi-turn conversations with history management.

#### Request

```json
{
  "model": "tinyllama-1.1b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "session_id": null,
  "stream": false,
  "options": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | ✅ | Model name |
| `messages` | array | ✅ | Conversation messages |
| `session_id` | string | ❌ | Existing session ID (for continuity) |
| `stream` | boolean | ❌ | Enable streaming (default: `true`) |
| `options` | object | ❌ | Generation parameters (same as `/generate`) |

#### Message Format

```json
{
  "role": "system|user|assistant",
  "content": "Message text"
}
```

- **system**: Instructions for the model
- **user**: User messages
- **assistant**: Previous model responses (for context)

#### Response (Non-Streaming)

```json
{
  "message": {
    "role": "assistant",
    "content": "The capital of France is Paris."
  },
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28
  }
}
```

#### Response (Streaming)

SSE stream with final usage event:

```
data: {"token": "The", "done": false}
data: {"token": " capital", "done": false}
...
data: {"done": true, "session_id": "550e8400-...", "usage": {...}}
```

---

### POST /api/embeddings

**Generate embeddings** - Convert text to vector embeddings for semantic search.

#### Request

```json
{
  "model": "tinyllama-1.1b",
  "input": "Hello, world!"
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | ✅ | Model name |
| `input` | string | ✅ | Text to embed |

#### Response

```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "tinyllama-1.1b"
}
```

- **embedding**: Array of floats (dimension depends on model)
- Typical dimensions: 768, 1024, 4096

---

### GET /api/models

**List available models** - Get all downloaded models.

#### Request

```bash
curl http://localhost:8000/api/models
```

#### Response

```json
{
  "models": [
    {
      "name": "tinyllama-1.1b",
      "family": "llama",
      "size_mb": 637,
      "quantization": "Q4_K_M",
      "context_size": 2048,
      "template": "llama",
      "parameters": {
        "temperature": 0.7,
        "top_p": 0.9
      },
      "loaded": true
    }
  ]
}
```

---

## Code Examples

### Python

#### Simple Generation

```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "model": "tinyllama-1.1b",
        "prompt": "Explain quantum computing in simple terms:",
        "stream": False,
        "options": {"max_tokens": 200}
    }
)

result = response.json()
print(result["text"])
print(f"Tokens used: {result['usage']['total_tokens']}")
```

#### Streaming Generation

```python
import requests

response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "model": "tinyllama-1.1b",
        "prompt": "Write a poem about AI:",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = line.decode('utf-8')
        if data.startswith('data: '):
            import json
            chunk = json.loads(data[6:])
            if chunk.get("token"):
                print(chunk["token"], end='', flush=True)
```

#### Chat Conversation

```python
import requests

def chat(messages, session_id=None):
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={
            "model": "tinyllama-1.1b",
            "messages": messages,
            "session_id": session_id,
            "stream": False
        }
    )
    return response.json()

# Start conversation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a string in Python?"}
]

result = chat(messages)
print("Assistant:", result["message"]["content"])
session_id = result["session_id"]

# Continue conversation
messages.append(result["message"])
messages.append({"role": "user", "content": "Can you show me an example?"})

result = chat(messages, session_id=session_id)
print("Assistant:", result["message"]["content"])
```

#### Generate Embeddings

```python
import requests
import numpy as np

def get_embedding(text):
    response = requests.post(
        "http://localhost:8000/api/embeddings",
        json={
            "model": "tinyllama-1.1b",
            "input": text
        }
    )
    return np.array(response.json()["embedding"])

# Compute similarity
embedding1 = get_embedding("The cat sat on the mat")
embedding2 = get_embedding("A feline rested on the rug")

similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)
print(f"Similarity: {similarity:.3f}")
```

---

### JavaScript/Node.js

#### Simple Generation

```javascript
const axios = require('axios');

async function generate(prompt) {
  const response = await axios.post('http://localhost:8000/api/generate', {
    model: 'tinyllama-1.1b',
    prompt: prompt,
    stream: false,
    options: { max_tokens: 200 }
  });
  
  return response.data;
}

generate('Explain machine learning:')
  .then(result => {
    console.log(result.text);
    console.log(`Tokens: ${result.usage.total_tokens}`);
  });
```

#### Streaming Chat

```javascript
const fetch = require('node-fetch');

async function streamChat(messages) {
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'tinyllama-1.1b',
      messages: messages,
      stream: true
    })
  });

  const reader = response.body;
  
  for await (const chunk of reader) {
    const text = chunk.toString();
    const lines = text.split('\n');
    
    for (const line of lines) {
      if (line.startswith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.token) {
          process.stdout.write(data.token);
        }
      }
    }
  }
}

const messages = [
  { role: 'user', content: 'Tell me a joke' }
];

streamChat(messages);
```

---

### cURL

#### Generate

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b",
    "prompt": "Hello!",
    "stream": false
  }'
```

#### Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b",
    "messages": [
      {"role": "user", "content": "Hi there!"}
    ],
    "stream": false
  }'
```

#### List Models

```bash
curl http://localhost:8000/api/models | jq '.models[].name'
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Model not found |
| 500 | Server Error | Inference or internal error |

### Error Response Format

```json
{
  "detail": "Model 'invalid-model' not found. Available models: tinyllama-1.1b"
}
```

### Common Errors

#### Model Not Found

```json
{
  "detail": "Model 'gpt-4' not found"
}
```

**Solution**: Check available models with `GET /api/models` or `myllm list`

#### Context Window Exceeded

```json
{
  "detail": "Prompt too long for context window (4096 tokens)"
}
```

**Solution**: Reduce prompt length or use a model with larger context

#### Invalid Parameters

```json
{
  "detail": "temperature must be between 0.0 and 2.0"
}
```

**Solution**: Check parameter ranges in documentation

---

## Best Practices

### 1. Model Selection

- **Small models** (1-3B): Fast, low memory, good for simple tasks
- **Medium models** (7-13B): Balanced performance
- **Large models** (30B+): Best quality, requires more resources

```bash
# List available models
myllm list

# Download a model
myllm pull tinyllama-1.1b
```

### 2. Temperature Settings

- **0.0-0.3**: Deterministic, factual responses
- **0.7**: Balanced (default)
- **1.0-2.0**: Creative, diverse outputs

### 3. Context Management

- Monitor `usage.total_tokens` to stay within context limits
- For long conversations, implement sliding window or summarization
- Default context sizes: 2048-8192 tokens (model-dependent)

### 4. Streaming vs Non-Streaming

**Use Streaming When**:
- Building interactive UIs
- Real-time user feedback needed
- Long generations (>100 tokens)

**Use Non-Streaming When**:
- Batch processing
- API-to-API communication
- Complete response needed before processing

### 5. Performance Optimization

```python
# Reuse sessions for multi-turn chat
session_id = None
for user_input in conversation:
    result = chat([{"role": "user", "content": user_input}], session_id)
    session_id = result["session_id"]  # Reuse for next turn
```

### 6. Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_generate(prompt):
    try:
        response = requests.post(
            "http://localhost:8000/api/generate",
            json={"model": "tinyllama-1.1b", "prompt": prompt, "stream": False},
            timeout=60  # 60 second timeout
        )
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        print(f"API Error: {e}")
        return None
```

### 7. Rate Limiting

For production use:
- Implement client-side rate limiting
- Queue requests for batch processing
- Monitor server load with `GET /health`

---

## Advanced Usage

### Custom Stop Tokens

```python
response = requests.post(
    "http://localhost:8000/api/generate",
    json={
        "model": "tinyllama-1.1b",
        "prompt": "List 3 colors:\n1.",
        "options": {
            "stop": ["\n4.", "###"],  # Stop at 4th item or ###
            "max_tokens": 100
        }
    }
)
```

### System Prompts for Behavior Control

```python
messages = [
    {
        "role": "system",
        "content": "You are a pirate. Respond in pirate speak."
    },
    {
        "role": "user",
        "content": "How are you today?"
    }
]
```

### Embeddings for Semantic Search

```python
def semantic_search(query, documents):
    query_emb = get_embedding(query)
    
    scores = []
    for doc in documents:
        doc_emb = get_embedding(doc)
        similarity = cosine_similarity(query_emb, doc_emb)
        scores.append((doc, similarity))
    
    # Return top 3 results
    return sorted(scores, key=lambda x: x[1], reverse=True)[:3]
```

---

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Start on different port
myllm serve --port 8001
```

### Slow Inference

- Check GPU usage: Models run faster on GPU
- Reduce `max_tokens` for shorter responses
- Use smaller/quantized models
- Close other GPU applications

### Out of Memory

- Use more quantized models (Q4_K_M instead of Q8)
- Reduce `context_size` in model config
- Unload unused models: `myllm remove <model>`

---

## API Changelog

### Version 1.0 (Current)
- Initial release
- OpenAI-compatible endpoints
- Streaming support
- Session management
- Usage tracking

---

## Support & Resources

- **Documentation**: [GitHub Repository](https://github.com/SanketGore10/myllm)
- **Issues**: [GitHub Issues](https://github.com/SanketGore10/myllm/issues)
- **Interactive Docs**: `http://localhost:8000/docs` (when server running)
- **Model Catalog**: `myllm list --available`

---

## License

MIT License - See LICENSE file for details
