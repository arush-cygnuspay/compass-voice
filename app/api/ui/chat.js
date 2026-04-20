const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const quickReplies = document.getElementById("quick-replies");
const chatStatus = document.getElementById("chat-status");

const AUTO_CHECK_INTERVAL_MS = 5000;

let sessionId = "";
let requestInFlight = false;
let autoCheckTimer = null;
let typingIndicator = null;
let lastAssistantSignature = "";

function newSessionId() {
  return `ui-${Date.now()}`;
}

function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}

function setStatus(text, tone = "default") {
  chatStatus.textContent = text;
  chatStatus.className = "status-pill";
  if (tone === "live") {
    chatStatus.classList.add("status-live");
  }
  if (tone === "pending") {
    chatStatus.classList.add("status-pending");
  }
}

function setComposerDisabled(disabled) {
  requestInFlight = disabled;
  input.disabled = disabled;
  sendBtn.disabled = disabled;
  sendBtn.textContent = disabled ? "Sending..." : "Send";

  Array.from(quickReplies.querySelectorAll("button")).forEach(button => {
    button.disabled = disabled;
  });
}

function createMessageRow(sender) {
  const row = document.createElement("div");
  row.className = `message-row ${sender}`;
  return row;
}

function addTextMessage(text, sender) {
  const row = createMessageRow(sender);
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  if (typingIndicator) {
    return;
  }

  typingIndicator = createMessageRow("bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  typingIndicator.appendChild(bubble);
  chatWindow.appendChild(typingIndicator);
  scrollToBottom();
}

function removeTypingIndicator() {
  if (!typingIndicator) {
    return;
  }

  typingIndicator.remove();
  typingIndicator = null;
}

function renderBotResponse(data, options = {}) {
  const { suppressDuplicate = false } = options;
  const links = Array.isArray(data.links) ? data.links : [];
  const signature = JSON.stringify({
    responseKey: data.response_key,
    response: data.response,
    links: links.map(link => link.url),
    state: data.state
  });

  if (suppressDuplicate && signature === lastAssistantSignature) {
    return;
  }

  lastAssistantSignature = signature;

  const row = createMessageRow("bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const card = document.createElement("div");
  card.className = "message-card";

  const text = document.createElement("div");
  text.textContent = data.response;
  card.appendChild(text);

  if (links.length > 0) {
    const linkStack = document.createElement("div");
    linkStack.className = "link-stack";

    links.forEach(link => {
      const anchor = document.createElement("a");
      anchor.className = "link-chip";
      anchor.href = link.url;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      anchor.textContent = link.label;
      linkStack.appendChild(anchor);
    });

    card.appendChild(linkStack);

    if (data.sms_phone_number) {
      const meta = document.createElement("div");
      meta.className = "link-meta";
      meta.textContent = `The secure link was also sent by SMS to ${data.sms_phone_number}.`;
      card.appendChild(meta);
    }
  }

  bubble.appendChild(card);
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  scrollToBottom();
}

let currentQuickReplyMode = "single";

function renderQuickReplies(items, mode = "single") {
  quickReplies.innerHTML = "";
  currentQuickReplyMode = mode;

  if (!Array.isArray(items) || items.length === 0) {
    return;
  }

  if (mode === "single") {
    items.forEach(item => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "quick-reply";
      button.textContent = item;
      button.disabled = requestInFlight;
      button.onclick = () => sendMessage(item);
      quickReplies.appendChild(button);
    });
    return;
  }

  // ── Multi-select mode: toggle options, send on Done/Skip ──
  items.forEach(item => {
    const isDone = item.toLowerCase() === "done";
    const isSkip = item.toLowerCase() === "skip";
    const isAction = isDone || isSkip;

    const button = document.createElement("button");
    button.type = "button";
    button.disabled = requestInFlight;

    if (isAction) {
      button.className = "quick-reply quick-reply-action" + (isDone ? " quick-reply-done" : "");
      button.textContent = item;
      button.onclick = () => sendMessage(item.toLowerCase());
    } else {
      button.className = "quick-reply quick-reply-option";
      button.textContent = item;
      button.dataset.selected = "false";
      button.onclick = () => {
        if (requestInFlight) return;
        // Send the option name directly — the FSM handles one selection at a time
        sendMessage(item);
      };
    }
    quickReplies.appendChild(button);
  });
}

function updateStatusFromResponse(data) {
  if (data.waiting_external) {
    if (data.response_key === "waiting_for_checkout_completion") {
      setStatus("Checkout is open. Finish the address and payment, and chat will keep checking automatically.", "pending");
      return;
    }
    setStatus("Payment is in progress. Chat will keep checking automatically.", "pending");
    return;
  }

  if (data.response_key === "order_completed") {
    const suffix = data.order_number ? ` Order #${data.order_number} confirmed.` : " Order confirmed.";
    setStatus(`Completed.${suffix}`, "live");
    return;
  }

  if (data.state === "WAITING_FOR_ORDER_TYPE") {
    setStatus("Choose pickup or delivery to begin.", "live");
    return;
  }

  setStatus("Live chat is ready for your order.", "live");
}

function stopAutoCheck() {
  if (!autoCheckTimer) {
    return;
  }

  window.clearInterval(autoCheckTimer);
  autoCheckTimer = null;
}

function startAutoCheck() {
  if (autoCheckTimer) {
    return;
  }

  autoCheckTimer = window.setInterval(() => {
    if (requestInFlight) {
      return;
    }
    sendTurn("__auto_payment_check__", { autoCheck: true });
  }, AUTO_CHECK_INTERVAL_MS);
}

function syncAutoCheck(data) {
  if (data.auto_check_recommended) {
    startAutoCheck();
    return;
  }

  stopAutoCheck();
}

async function sendTurn(text, options = {}) {
  const { autoCheck = false, suppressUser = false, suppressDuplicateBot = false } = options;

  if (requestInFlight) {
    return;
  }

  if (!autoCheck && !suppressUser) {
    addTextMessage(text, "user");
  }

  setComposerDisabled(true);
  addTypingIndicator();

  try {
    const response = await fetch("/test/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        text
      })
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    removeTypingIndicator();
    renderBotResponse(data, { suppressDuplicate: suppressDuplicateBot || autoCheck });
    renderQuickReplies(data.quick_replies, data.quick_reply_mode || "single");
    updateStatusFromResponse(data);
    syncAutoCheck(data);
  } catch (error) {
    removeTypingIndicator();
    stopAutoCheck();
    addTextMessage("Something went wrong while contacting the chat service. Please try again.", "system");
    setStatus("Connection issue. You can send the message again.", "pending");
  } finally {
    setComposerDisabled(false);
    input.focus();
  }
}

async function sendMessage(prefilledText = null) {
  const text = (prefilledText ?? input.value).trim();
  if (!text) {
    return;
  }

  input.value = "";
  await sendTurn(text);
}

async function bootstrapChat() {
  sessionId = newSessionId();
  lastAssistantSignature = "";
  stopAutoCheck();
  chatWindow.innerHTML = "";
  quickReplies.innerHTML = "";
  setStatus("Starting a new order...", "pending");
  await sendTurn("", { suppressUser: true, suppressDuplicateBot: false });
}

newChatBtn.onclick = () => {
  bootstrapChat();
};

sendBtn.onclick = () => {
  sendMessage();
};

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

bootstrapChat();
