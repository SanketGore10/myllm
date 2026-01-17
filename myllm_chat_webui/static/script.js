let sessionId = crypto.randomUUID();
let currentModel = null;

window.onload = async () => {
  await loadModels();
};

async function loadModels() {
  const res = await fetch("http://127.0.0.1:8000/api/models");
  const data = await res.json();

  const select = document.getElementById("model-select");
  data.models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    select.appendChild(opt);
  });

  currentModel = data.models[0]?.name;
}

function newChat() {
  sessionId = crypto.randomUUID();
  document.getElementById("chat-box").innerHTML = "";
}

function addMessage(text, cls) {
  const box = document.getElementById("chat-box");
  const div = document.createElement("div");
  div.className = `message ${cls}`;
  div.innerHTML = marked.parse(text);
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  hljs.highlightAll();
  return div;
}

async function send() {
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text) return;

  input.value = "";

  addMessage(text, "user");
  const botDiv = addMessage("", "bot");

  const res = await fetch("http://127.0.0.1:8000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: currentModel,
      session_id: sessionId,
      stream: true,
      messages: [{ role: "user", content: text }]
    })
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let full = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;

      const data = JSON.parse(line.slice(6));

      if (data.token) {
        full += data.token;
        botDiv.innerHTML = marked.parse(full);
        boxScroll();
      }
    }
  }
}

function boxScroll() {
  const box = document.getElementById("chat-box");
  box.scrollTop = box.scrollHeight;
}
