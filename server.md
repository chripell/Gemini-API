# `server.py` - OpenAI-Compatible API Server for Gemini WebAPI

`server.py` provides a lightweight, high-performance, standard **OpenAI-compatible HTTP API server** backed by the `gemini_webapi` library.

It allows you to use your Google Gemini Web account with any tool, library, or Web UI designed for OpenAI's API (such as the official `openai` Python SDK, LangChain, LlamaIndex, Open WebUI, LibreChat, Continue, Cursor, and more) with **zero extra framework dependencies** (pure Python standard library `asyncio` server).

---

## Key Features

- **Standard OpenAI Endpoints**:
  - `POST /openai/v1/chat/completions` & `POST /v1/chat/completions` (Non-streaming & Streaming SSE).
  - `GET /openai/v1/models` & `GET /v1/models` (Discovered models list).
  - `GET /openai/v1/models/{model}` & `GET /v1/models/{model}` (Model information).
  - `POST /openai/v1/chat/reset` & `POST /v1/chat/reset` (Reset the shared chat session).
  - `GET /health` & `GET /` (Server and account health check).
  - `GET /openai/v1/account` & `GET /v1/account` (Account diagnostics, quotas, and compute credits).
- **Shared Chat Session**:
  - A single `ChatSession` is shared across all connecting clients and lazily opened upon the first request.
  - Concurrency is managed with an `asyncio.Lock` to ensure safe, serialized conversation state.
- **Full Streaming & Reasoning Support**:
  - Incremental token streaming (`stream: true`) formatted as Server-Sent Events (SSE).
  - Thinking / reasoning process streamed in `choices[0].delta.reasoning_content` (and included in non-streaming responses).
- **Multimodal & Image Support**:
  - Supports text and images (Base64 data URLs `data:image/...;base64,...` or remote URLs).
  - Formats generated and referenced web images into clean markdown in responses.
- **Startup Diagnostics**:
  - Automatically probes and displays account status, plan tier, AI compute credits, quota limits, abuse status, and discovered models at launch.
- **Zero Extra Dependencies**:
  - Built using Python's built-in `asyncio.start_server`, eliminating the need for FastAPI, Uvicorn, or Flask.
- **Automatic Cookie Persistence**:
  - Saves rotated and updated session cookies back to your cookies JSON file on exit unless `--no-persist` is given.

---

## Command-Line Usage

```bash
python3 server.py [OPTIONS]
```

### Options Reference

| Option | Default | Description |
| :--- | :--- | :--- |
| `--addr`, `--host` | `127.0.0.1` | IP address or hostname to bind. Use `0.0.0.0` for all interfaces. |
| `--port`, `-p` | `4981` | TCP port number to listen on. |
| `--temp`, `--temporary` | `False` | Use temporary chat session (turns will not appear in Gemini web history). |
| `--simple` | `False` | Prepend concise ASCII plain-text instruction to the first prompt of the conversation. |
| `--cookies`, `--cookies-json`, `--cookes` | `None` | Path to JSON cookies file exported from browser. |
| `--model`, `-m` | Account default | Default model name or ID (e.g. `gemini-flash`, `gemini-pro`). |
| `--gem` | `None` | Gem ID or name to use as custom instructions / persona. |
| `--account-index` | `None` | Google account index (for multi-account cookies). |
| `--no-persist` | `False` | Do not write updated session cookies back to disk on exit. |
| `--no-auto-refresh` | `False` | Disable automatic background cookie refresh. |
| `--proxy` | `HTTPS_PROXY` | HTTP/HTTPS proxy URL. |
| `--request-timeout` | `300` | Per-request timeout in seconds. |
| `--skip-verify` | `False` | Skip SSL certificate verification. |
| `--verbose`, `-v` | `False` | Enable verbose / debug logging. |

### Environment Variables

Alternatively, you can provide cookies via environment variables instead of `--cookies`:
- `GEMINI_SECURE_1PSID` (or `SECURE_1PSID`): `__Secure-1PSID` cookie value (required).
- `GEMINI_SECURE_1PSIDTS` (or `SECURE_1PSIDTS`): `__Secure-1PSIDTS` cookie value (optional but recommended).
- `HTTPS_PROXY` / `HTTP_PROXY`: Proxy URL.

---

## Startup Output Example

When launching `server.py`, you will see account diagnostics and model discovery statistics:

```text
============================================================
 Gemini WebAPI - OpenAI Compatible Server
============================================================
 Account Status: AVAILABLE (0) - Available to generate content
 Abuse Status:   clean
 Plan Tier:      Gemini Advanced (1)
   - 5h       0% used, 100 credits left
   - weekly   2% used, 980 credits left
   - credits: 500 AI credits remaining
 Quotas:
   - Gemini Flash               100/100 remaining (0% used)
   - Gemini Pro                 50/50 remaining (0% used)
   - extra features             ok (0% used)

 Discovered Models (2):
   Name          Display       ID
   --------------------------------------------------
   gemini-flash  Gemini Flash  models/gemini-flash
   gemini-pro    Gemini Pro    models/gemini-pro
============================================================
[Server] Listening on ('127.0.0.1', 4981)
[Server] Temporary chat: False
[Server] Endpoint ready: http://127.0.0.1:4981/openai/v1/chat/completions
[Server] (Press Ctrl+C to stop)
```

---

## API Endpoints

### 1. Chat Completions (`POST /openai/v1/chat/completions` or `POST /v1/chat/completions`)

#### Non-Streaming Example

```bash
curl -X POST http://localhost:4981/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1724526145,
  "model": "gemini-flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 8,
    "total_tokens": 10
  }
}
```

#### Streaming Example (`stream: true`)

```bash
curl -N -X POST http://localhost:4981/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash",
    "messages": [
      {"role": "user", "content": "Tell me a short poem."}
    ],
    "stream": true
  }'
```

**SSE Chunks Output:**
```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1724526145,"model":"gemini-flash","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1724526145,"model":"gemini-flash","choices":[{"index":0,"delta":{"content":"The ocean waves "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1724526145,"model":"gemini-flash","choices":[{"index":0,"delta":{"content":"embrace the shore...\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1724526145,"model":"gemini-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":6,"completion_tokens":12,"total_tokens":18}}

data: [DONE]
```

#### Multimodal / Image Input Example

You can pass base64 image data URLs or remote image URLs inside `messages`:

```bash
curl -X POST http://localhost:4981/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is depicted in this image?"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}}
        ]
      }
    ]
  }'
```

---

### 2. List Models (`GET /openai/v1/models` or `GET /v1/models`)

```bash
curl http://localhost:4981/openai/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gemini-flash",
      "object": "model",
      "created": 1724526145,
      "owned_by": "google",
      "permission": [],
      "root": "models/gemini-flash",
      "parent": null
    },
    {
      "id": "gemini-pro",
      "object": "model",
      "created": 1724526145,
      "owned_by": "google",
      "permission": [],
      "root": "models/gemini-pro",
      "parent": null
    }
  ]
}
```

---

### 3. Model Information (`GET /openai/v1/models/{model}` or `GET /v1/models/{model}`)

```bash
curl http://localhost:4981/openai/v1/models/gemini-flash
```

---

### 4. Reset Shared Chat Session (`POST /openai/v1/chat/reset` or `POST /v1/chat/reset`)

Clears the shared chat session so subsequent requests start with a fresh conversation context:

```bash
curl -X POST http://localhost:4981/openai/v1/chat/reset
```

**Response:**
```json
{
  "status": "ok",
  "message": "Shared chat session reset"
}
```

---

### 5. Health & Diagnostics (`GET /health` and `GET /openai/v1/account`)

```bash
# Basic Health Probe
curl http://localhost:4981/health

# Detailed Account Diagnostics (Quotas, Abuse status, Tier, Credits)
curl http://localhost:4981/openai/v1/account
```

---

## Client Integration Examples

### 1. Official OpenAI Python SDK

```python
from openai import OpenAI

# Point client to local Gemini server
client = OpenAI(
    base_url="http://localhost:4981/openai/v1",
    api_key="not-needed",  # Any string
)

# 1. Non-streaming completion
response = client.chat.completions.create(
    model="gemini-flash",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print("Response:", response.choices[0].message.content)

# 2. Streaming completion
stream = client.chat.completions.create(
    model="gemini-flash",
    messages=[{"role": "user", "content": "Count from 1 to 5"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 2. LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:4981/v1",
    api_key="not-needed",
    model="gemini-flash",
)

response = llm.invoke("What are the 3 laws of robotics?")
print(response.content)
```

### 3. Open WebUI / LibreChat / ChatGPT-Next-Web / Continue

In the application's OpenAI API settings:
- **Base URL / API Host**: `http://localhost:4981/v1` (or `http://localhost:4981/openai/v1`)
- **API Key**: `any-key` (can be any non-empty string)
- **Model**: `gemini-flash` or `gemini-pro`

---

## Architecture & Shared Session Behavior

1. **Lazy Initialization**:
   The `ChatSession` is initialized only when the first chat completion request arrives. This allows the server to start instantaneously while verifying credentials and model availability.
2. **Session Persistence**:
   All connecting clients participate in a single shared conversation context. If you send turn 1 and then turn 2, Gemini maintains the chat context server-side.
3. **Session Reset**:
   If you wish to wipe the conversation and start fresh, invoke `POST /openai/v1/chat/reset` or restart `server.py`.
4. **Concurrency**:
   An asynchronous mutex lock (`asyncio.Lock`) serializes turns in the shared chat session, preventing conflicting multi-turn race conditions.
5. **Logging**:
   - **Successful requests** emit a concise single log line to `stdout`:
     ```text
     [2026-08-24 21:06:51] 127.0.0.1 "POST /openai/v1/chat/completions HTTP/1.1" 200 OK - model=gemini-flash, tokens=1/4 (0.42s)
     ```
   - **Errors** output descriptive details and stack traces to `stderr` to facilitate debugging.
6. **Simple / Concise Mode (`--simple`)**:
   When launched with `--simple`, the server automatically prepends the following plain ASCII directives to the first prompt sent to the shared chat session:
   > *Be concise in your response. Answer only what has been asked. Avoid additional text, links or follow-up questions. Do not use any special markup unless explicitly stated in the question, by default use only simple ASCII characters.*

   This instructs Gemini to provide direct, clean answers without conversational filler or complex markup. If the session is reset via `/chat/reset`, the directive is applied again to the first turn of the new session.
