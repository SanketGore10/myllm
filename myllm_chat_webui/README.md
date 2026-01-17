# MyLLM Chat WebUI

A modern, beautiful web interface for chatting with MyLLM models locally.

## Features

- ✨ **Modern UI** with glassmorphism and smooth animations
- 🌓 **Dark/Light theme** toggle
- 📡 **Real-time streaming** responses
- 🤖 **Multiple model support** with live model switching
- 💬 **Multi-session chat** management
- 📊 **Token usage tracking**
- 📋 **Copy & regenerate** messages
- 🎨 **Syntax highlighting** for code blocks
- 📱 **Responsive design** for mobile

## Prerequisites

- MyLLM server running on `http://localhost:8000`
- Python 3.8+
- At least one model installed (use `myllm pull <model-name>`)

## Installation

```bash
# Navigate to webui directory
cd myllm_chat_webui

# Install dependencies
pip install fastapi uvicorn requests python-multipart jinja2

# Or if you have requirements.txt
pip install -r requirements.txt
```

## Running

### Start MyLLM Server (Terminal 1)

```bash
cd myllm
myllm serve
```

### Start WebUI (Terminal 2)

```bash
cd myllm_chat_webui
python main.py
```

The WebUI will be available at: **http://localhost:5000**

## Usage

1. **Select a Model**: Choose from available models in the dropdown
2. **Start Chatting**: Type your message and press Ctrl+Enter or click Send
3. **Stream Responses**: Watch tokens appear in real-time
4. **Create New Chats**: Click "✨ New Chat" to start fresh
5. **Toggle Theme**: Click 🌓 to switch between dark/light modes

## Configuration

Edit `main.py` to customize:

```python
MYLLM_BASE_URL = "http://localhost:8000"  # MyLLM server URL

# In __main__ section:
uvicorn.run(app, host="0.0.0.0", port=5000)  # WebUI port
```

## Project Structure

```
myllm_chat_webui/
├── main.py              # FastAPI backend
├── static/
│   ├── style.css       # Modern styling
│   └── script.js       # Frontend logic
└── templates/
    └── index.html      # HTML template
```

## API Endpoints

- `GET /` - Main chat interface
- `GET /api/models` - List available models
- `POST /chat` - Send message (supports streaming)
- `POST /new_chat` - Create new chat session
- `DELETE /chat/{session_id}` - Delete chat session

## Troubleshooting

### "Cannot connect to MyLLM server"

- Ensure MyLLM server is running: `myllm serve`
- Check server is on port 8000: `curl http://localhost:8000/health`

### "No models found"

- Download a model: `myllm pull [model-name]`
- Verify: `myllm list`

### WebUI not loading

- Check port 5000 is available
- Try different port: Edit `main.py` uvicorn.run port=5001

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload --port 5000
```

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Vanilla JavaScript, Marked.js, Highlight.js
- **Styling**: Custom CSS with glassmorphism
- **Streaming**: Server-Sent Events (SSE)

## License

MIT License - Same as MyLLM parent project
