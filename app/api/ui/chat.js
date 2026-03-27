const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");

// Generate a new session id
function newSessionId() {
  return "ui-" + Date.now();
}

let sessionId = newSessionId();

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}


async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  const response = await fetch("/test/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      text: text
    })
  });

  const data = await response.json();
  addMessage(data.response, "bot");
}

// ---- New Chat ----
newChatBtn.onclick = () => {
  sessionId = newSessionId();
  chatWindow.innerHTML = "";
  addMessage("🆕 New chat started.", "bot");
};

sendBtn.onclick = sendMessage;
input.addEventListener("keydown", e => {
  if (e.key === "Enter") sendMessage();
});
