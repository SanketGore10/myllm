from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import json
import uuid

app = FastAPI(title="MyLLM ChatGPT UI")

# -----------------------------
# Config
# -----------------------------
MYLLM_BASE = "http://127.0.0.1:8000"
MYLLM_MODELS = f"{MYLLM_BASE}/api/models"
MYLLM_CHAT = f"{MYLLM_BASE}/api/chat"

# -----------------------------
# Static
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# In-memory chat sessions
# -----------------------------
sessions = {}

# -----------------------------
# Pages
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# Models proxy
# -----------------------------
@app.get("/api/models")
async def get_models():
    try:
        r = requests.get(MYLLM_MODELS, timeout=5)
        return r.json()
    except Exception as e:
        return {"models": [], "error": str(e)}


# -----------------------------
# New chat
# -----------------------------
@app.post("/new_chat")
async def new_chat():
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid}


# -----------------------------
# Chat proxy (streaming)
# -----------------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()

    message = data.get("message", "").strip()
    session_id = data.get("session_id")
    model = data.get("model")
    stream = data.get("stream", True)

    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    if session_id not in sessions:
        sessions[session_id] = []

    # build conversation
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    for turn in sessions[session_id]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})

    messages.append({"role": "user", "content": message})

    # -----------------------------
    # STREAMING
    # -----------------------------
    if stream:
        def event_stream():
            response = requests.post(
                MYLLM_CHAT,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=120
            )

            full_text = ""

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8")

                if not decoded.startswith("data: "):
                    continue

                payload = decoded[6:]

                try:
                    chunk = json.loads(payload)
                except:
                    continue

                if chunk.get("token"):
                    full_text += chunk["token"]
                    yield f"data: {json.dumps(chunk)}\n\n"

                if chunk.get("done"):
                    sessions[session_id].append({
                        "user": message,
                        "bot": full_text
                    })
                    yield f"data: {json.dumps(chunk)}\n\n"
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # -----------------------------
    # NON-STREAM
    # -----------------------------
    else:
        r = requests.post(
            MYLLM_CHAT,
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        data = r.json()
        bot = data["message"]["content"]

        sessions[session_id].append({
            "user": message,
            "bot": bot
        })

        return {"response": bot}


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
