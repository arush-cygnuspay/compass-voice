# Restaurant Voice AI

**Real-time AI voice ordering system for restaurants using Twilio Media Streams, Deepgram STT/TTS, custom NLU, Redis-backed sessions, and deterministic conversational orchestration.**

Restaurant Voice AI handles multi-turn restaurant ordering conversations over live phone calls while maintaining explicit control over conversation state, cart operations, checkout, and payment flows.

Rather than relying on an LLM to control the conversation, the system separates **language understanding** from **business logic**: machine-learning-based NLU interprets the caller, while deterministic state-machine logic decides what happens next.

---

## Why This Project

Voice ordering is more difficult than a typical chatbot because the system has to operate under real-time telephony constraints while maintaining reliable transactional state.

The system addresses several engineering problems:

- streaming audio between Twilio and the application over WebSockets
- real-time speech recognition and speech synthesis
- intent and slot extraction from conversational speech
- deterministic multi-turn conversation control
- stateful cart and order management
- Redis-backed session persistence
- checkout, SMS, and payment-link workflows
- human-agent transfer paths
- latency and NLU observability

The goal is to combine AI-based language understanding with predictable application behavior suitable for transactional workflows.

---

## Architecture

```text
                       ┌─────────────────────┐
                       │   Restaurant Caller │
                       └──────────┬──────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Twilio      │
                         │  Media Streams  │
                         └────────┬────────┘
                                  │ WebSocket audio
                                  ▼
                     ┌────────────────────────┐
                     │        FastAPI         │
                     │  Voice Stream Server   │
                     └───────────┬────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
                 ▼                                ▼
        ┌─────────────────┐              ┌─────────────────┐
        │ Deepgram STT    │              │ Deepgram TTS    │
        │ Speech → Text   │              │ Text → Speech   │
        └────────┬────────┘              └────────▲────────┘
                 │                                │
                 ▼                                │
        ┌─────────────────┐                       │
        │   NLU Pipeline  │                       │
        │ Intent + Slots  │                       │
        └────────┬────────┘                       │
                 │                                │
                 ▼                                │
        ┌──────────────────────┐                  │
        │      Turn Engine     │                  │
        └──────────┬───────────┘                  │
                   │                              │
                   ▼                              │
        ┌──────────────────────┐                  │
        │ Deterministic State  │                  │
        │ Machine + Router     │                  │
        └──────────┬───────────┘                  │
                   │                              │
                   ▼                              │
        ┌──────────────────────┐                  │
        │ State-Specific       │                  │
        │ Handlers             │                  │
        └──────────┬───────────┘                  │
                   │                              │
          ┌────────┴─────────┐                    │
          ▼                  ▼                    │
   ┌─────────────┐    ┌──────────────┐            │
   │ Cart / Menu │    │ Checkout /   │            │
   │ Operations  │    │ Payments/SMS │            │
   └──────┬──────┘    └──────┬───────┘            │
          └──────────┬────────┘                    │
                     ▼                             │
             ┌───────────────┐                     │
             │   Response    │─────────────────────┘
             │    Builder    │
             └───────────────┘
```

### Conversation Flow

```text
Caller audio
    ↓
Streaming speech-to-text
    ↓
Intent + slot extraction
    ↓
Turn engine
    ↓
Deterministic state routing
    ↓
State-specific business logic
    ↓
Response generation
    ↓
Streaming text-to-speech
    ↓
Audio returned to caller
```

---

## Engineering Highlights

### Real-Time Voice Pipeline

Twilio Media Streams deliver live call audio to the FastAPI application over WebSockets.

The application processes Twilio-compatible μ-law 8 kHz audio and integrates streaming Deepgram speech-to-text and text-to-speech services to support low-latency conversational turns.

### Custom NLU

The NLU layer identifies user intent and extracts information needed by the ordering workflow.

Instead of allowing the language model or classifier to directly control application behavior, NLU output is passed into explicit routing and business-logic components.

This separation makes conversational behavior more predictable and easier to debug.

### Deterministic Conversation Orchestration

Conversation state is controlled through an explicit state machine.

Examples of stateful interactions include:

```text
select item
→ resolve variation
→ select size
→ select side
→ choose quantity
→ confirm item
→ update cart
→ continue ordering / checkout
```

Each state is handled independently, reducing the risk of unexpected transitions during transactional conversations.

### Stateful Sessions

Redis-backed session storage maintains conversation context across turns, including order state and cart information.

This allows the voice layer, NLU pipeline, and ordering logic to remain separated while sharing consistent session state.

### Checkout and Payment Flows

The system includes application flows for:

- cart management
- checkout
- SMS interactions
- payment links
- payment-status waiting states
- order confirmation
- human-agent transfer

Transactional states are modeled explicitly rather than inferred conversationally.

### Observability

Runtime instrumentation includes:

- turn-level latency tracking
- NLU decision logging
- speech-processing diagnostics
- configurable debug logging

Runtime-generated logs are excluded from source control and can be written to a dedicated `runtime-logs/` directory.

---

## Browser Demo

The repository also contains a lightweight browser interface for exercising the same conversation engine without making a phone call.

Available application routes include:

```text
/ui                     Browser demo
POST /test/chat         Test chat API
POST /voice             Twilio voice webhook
/ws/twilio-media        Twilio Media Streams WebSocket
```

This provides a faster way to inspect conversation behavior independently of the telephony layer.

---

## Technology Stack

| Area | Technologies |
|---|---|
| Backend | Python, FastAPI, WebSockets |
| Telephony | Twilio, Twilio Media Streams |
| Speech | Deepgram STT, Deepgram Aura TTS |
| NLP / ML | PyTorch, Transformers, custom intent and slot inference |
| Conversation Control | Deterministic state machine, handler-based routing |
| Session State | Redis |
| Infrastructure | Docker, Docker Compose |
| Deployment | GitHub Actions, Docker Hub, SSH-based deployment |
| Observability | Structured latency and NLU logging |

---

## Project Structure

```text
restaurant-voice-ai/
├── app/
│   ├── api/              # FastAPI endpoints and voice streaming
│   ├── bootstrap/        # Runtime dependency wiring
│   ├── cart/             # Cart operations
│   ├── core/             # Turn engine and response orchestration
│   ├── data/             # Synthetic demo restaurant data
│   ├── logging/          # Latency and NLU logging
│   ├── menu/             # Menu loading and resolution
│   ├── ml/               # Intent and slot inference
│   ├── realtime/         # Streaming STT/TTS and realtime controllers
│   ├── responses/        # Response construction
│   ├── services/         # SMS, checkout, payment, and call integrations
│   ├── session/          # Session persistence
│   ├── state_machine/    # Conversation states, routing, and handlers
│   ├── static/
│   └── utils/
│
├── tests/                # Automated tests for core conversation flows
├── .env.example
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.12
- Redis
- Twilio account for live phone testing
- Deepgram API key

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example configuration:

```bash
cp .env.example .env
```

Configure the required credentials:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key

TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

PUBLIC_WSS_BASE_URL=https://your-public-host.example

REDIS_HOST=localhost
REDIS_PORT=6379
```

Additional runtime options are documented in `.env.example`.

### Run with Docker Compose

```bash
docker compose up --build
```

The Compose configuration starts the application and Redis services.

### Run Locally

```bash
uvicorn app.api.voice_stream_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

A public HTTPS/WSS endpoint is required when connecting Twilio Media Streams to a local development environment.

---

## Tests

The repository includes automated tests covering core conversation and state-machine behavior.

Tests can be executed with:

```bash
pytest
```

The current portfolio cleanup focuses on repository structure and presentation; the test suite has not been revalidated as part of that cleanup.

---

## Deployment

The repository includes a GitHub Actions deployment workflow demonstrating:

```text
GitHub Actions
    ↓
Docker Buildx
    ↓
Docker image
    ↓
Docker Hub
    ↓
SSH deployment
    ↓
Docker Compose
```

Deployment is intentionally **manual-triggered** through `workflow_dispatch` rather than running automatically on every push to `main`.

Production credentials and server configuration are supplied through GitHub Actions secrets.

---

## Design Principle

The central design decision in this project is:

> **Use machine learning to understand the conversation, but deterministic software to control transactional behavior.**

This keeps AI where it is useful—interpreting natural language—while keeping ordering, payment, state transitions, and side effects explicit and predictable.