let currentSession = null;
let currentModel = null;
let isGenerating = false;

window.onload = async () => {
  await loadModels();
  createNewChat();
  setupInputHandlers();
};

async function loadModels() {
  try {
    const res = await fetch("http://127.0.0.1:8000/api/models");
    const data = await res.json();

    const select = document.getElementById("model-select");
    select.innerHTML = "";

    if (data.models && data.models.length > 0) {
      data.models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.name;
        select.appendChild(opt);
      });

      currentModel = data.models[0]?.name;
      updateServerStatus(true);
    } else {
      select.innerHTML = '<option value="">No models found</option>';
      updateServerStatus(false, 'No models');
    }
  } catch (e) {
    console.error("Failed to load models:", e);
    document.getElementById("model-select").innerHTML = '<option value="">Server offline</option>';
    updateServerStatus(false, 'Disconnected');
  }
}

function changeModel() {
  currentModel = document.getElementById("model-select").value;
}

function updateServerStatus(online, message = null) {
  const statusEl = document.getElementById("server-status");
  const dot = statusEl.querySelector(".status-dot");
  const text = statusEl.querySelector(".status-text");

  if (online) {
    dot.classList.add("online");
    text.textContent = "Connected";
  } else {
    dot.classList.remove("online");
    text.textContent = message || "Disconnected";
  }
}

function createNewChat() {
  currentSession = crypto.randomUUID();
  document.getElementById("chat-box").innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">🤖</div>
      <h2>Welcome to MyLLM</h2>
      <p>Chat locally with your own models.</p>
    </div>
  `;
  document.getElementById("chat-title-text").textContent = "New chat";
}

function clearChat() {
  if (confirm("Clear this chat?")) {
    createNewChat();
  }
}

function addMessage(text, sender) {
  const box = document.getElementById("chat-box");

  // Remove welcome message
  const welcome = box.querySelector(".welcome-message");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = `message ${sender}-message`;

  if (sender === "bot") {
    div.innerHTML = marked.parse(text);
    hljs.highlightAll();
  } else {
    div.textContent = text;
  }

  box.appendChild(div);
  scrollToBottom();
  return div;
}

async function sendMessage() {
  console.log("Sending message");

  const input = document.getElementById("user-input");
  const text = input.value.trim();

  console.log("Message:", text);

  if (!text || !currentSession || isGenerating) return;

  input.value = "";
  updateCharCount();

  addMessage(text, "user");

  isGenerating = true;
  showTyping();
  document.getElementById("stop-btn").classList.remove("hidden");

  try {
    const res = await fetch("http://127.0.0.1:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: currentModel,
        session_id: currentSession,
        stream: true,
        messages: [{ role: "user", content: text }]
      })
    });

    console.log("Response:", res);

    hideTyping();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let full = "";
    let botDiv = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;

        try {
          const data = JSON.parse(line.slice(6));

          if (data.token) {
            full += data.token;

            if (!botDiv) {
              botDiv = addMessage("", "bot");
            }

            botDiv.innerHTML = marked.parse(full);
            hljs.highlightAll();
            scrollToBottom();
          }

          if (data.done) {
            if (data.usage) {
              updateUsageInfo(data.usage);
            }
            break;
          }
        } catch (e) {
          // Skip invalid JSON
        }
      }
    }
  } catch (e) {
    console.error("Send error:", e);
    hideTyping();
    addMessage("❌ Error: Cannot connect to MyLLM server. Is it running on port 8000?", "bot");
  } finally {
    isGenerating = false;
    document.getElementById("stop-btn").classList.add("hidden");
  }
}

function stopGeneration() {
  isGenerating = false;
  hideTyping();
  document.getElementById("stop-btn").classList.add("hidden");
}

function showTyping() {
  document.getElementById("typing-indicator").classList.remove("hidden");
  scrollToBottom();
}

function hideTyping() {
  document.getElementById("typing-indicator").classList.add("hidden");
}

function scrollToBottom() {
  const box = document.getElementById("chat-box");
  box.scrollTop = box.scrollHeight;
}

function setupInputHandlers() {
  const input = document.getElementById("user-input");

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
    updateCharCount();
  });

  updateCharCount();
}

function handleKeyPress(e) {
  if (e.key === "Enter" && e.ctrlKey) {
    e.preventDefault();
    sendMessage();
  }
}

function updateCharCount() {
  const input = document.getElementById("user-input");
  const count = document.getElementById("char-count");
  count.textContent = `${input.value.length} characters`;
}

function updateUsageInfo(usage) {
  const info = document.getElementById("usage-info");
  if (usage) {
    info.textContent = `Tokens: ${usage.total_tokens}`;
  } else {
    info.textContent = "";
  }
}
