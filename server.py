#!/usr/bin/env python3
"""OpenAI-compatible API server backed by gemini-webapi."""

import argparse
import asyncio
import base64
import contextlib
import io
import json
import os
import re
import signal
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# Run straight from a source checkout, without the package having to be installed
if (_src := str(Path(__file__).resolve().parent / "src")) not in sys.path:
    sys.path.insert(0, _src)

from gemini_webapi import GeminiClient, logger, set_log_level
from gemini_webapi.exceptions import APIError, AuthError, GeminiError
from gemini_webapi.types.image import GeneratedImage, WebImage

# ---------------------------------------------------------------------------
# region - Cookie helpers (compatible with cli.py)
# ---------------------------------------------------------------------------


def _parse_expiry(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        with contextlib.suppress(ValueError):
            return int(float(raw))
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return int(dt.timestamp())
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp())
        except Exception:
            return None
    return None


def _load_cookies_with_meta(path: str) -> tuple[dict[str, str], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cookies: dict[str, str] = {}
    meta: dict[str, Any] = {}

    def _upsert(name: Any, value: Any, expires_raw: Any = None) -> None:
        if not isinstance(name, str) or not name:
            return
        if not isinstance(value, str) or not value:
            return
        cookies[name] = value
        exp = _parse_expiry(expires_raw)
        meta[name] = {
            "expires_raw": expires_raw,
            "expires_epoch": exp,
            "expires_iso": (
                datetime.fromtimestamp(exp, tz=UTC).isoformat().replace("+00:00", "Z")
                if exp is not None
                else None
            ),
        }

    def _handle_obj(item: dict[str, Any]) -> None:
        name = item.get("name")
        value = item.get("value")
        expires_raw = (
            item.get("expirationDate")
            or item.get("expires")
            or item.get("expiry")
            or item.get("expiresDate")
        )
        _upsert(name, value, expires_raw=expires_raw)

    # Flat {name: value}
    if isinstance(data, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        for k, v in data.items():
            _upsert(k, v)
        return cookies, meta
    # {"cookies": {name: value}}
    if isinstance(data, dict) and isinstance(data.get("cookies"), dict):
        inner = data["cookies"]
        if all(isinstance(v, str) for v in inner.values()):
            for k, v in inner.items():
                _upsert(k, v)
            return cookies, meta
    # {"cookies": [{name, value}, ...]}
    if isinstance(data, dict) and isinstance(data.get("cookies"), list):
        for item in data["cookies"]:
            if isinstance(item, dict):
                _handle_obj(item)
        if cookies:
            return cookies, meta
    # [{name, value}, ...]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _handle_obj(item)
        if cookies:
            return cookies, meta

    raise ValueError(f"Unsupported cookie format in {path}")


def _persist_cookies(
    cookies_json_path: str,
    original: dict[str, str],
    client_cookies: Any,
    verbose: bool = False,
) -> None:
    merged = dict(original)
    with contextlib.suppress(Exception):
        for cookie in client_cookies.jar:
            name = getattr(cookie, "name", None)
            value = getattr(cookie, "value", None)
            if isinstance(name, str) and isinstance(value, str) and value:
                merged[name] = value
    if merged == original:
        return
    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "cookies": dict(sorted(merged.items())),
    }
    Path(cookies_json_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        changed = [k for k in merged if merged[k] != original.get(k)]
        sys.stdout.write(f"[Server] Persisted cookies ({', '.join(changed)})\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# region - Client Initialization & Diagnostic Printing
# ---------------------------------------------------------------------------


def _build_client(args: argparse.Namespace) -> tuple[GeminiClient, dict[str, str]]:
    json_cookies: dict[str, str] = {}
    cookie_path = args.cookies_json or args.cookies or args.cookes
    if cookie_path:
        json_cookies, _ = _load_cookies_with_meta(cookie_path)

    psid = (
        json_cookies.get("__Secure-1PSID")
        or os.getenv("GEMINI_SECURE_1PSID")
        or os.getenv("SECURE_1PSID")
    )
    psidts = (
        json_cookies.get("__Secure-1PSIDTS")
        or os.getenv("GEMINI_SECURE_1PSIDTS")
        or os.getenv("SECURE_1PSIDTS")
    )

    if not psid:
        raise SystemExit(
            "Error: Missing __Secure-1PSID. Pass cookie jar via --cookies <path> or set GEMINI_SECURE_1PSID."
        )
    if not psidts:
        sys.stderr.write("Warning: __Secure-1PSIDTS cookie not found.\n")

    extra = {
        k: v
        for k, v in json_cookies.items()
        if k not in {"__Secure-1PSID", "__Secure-1PSIDTS"}
    }

    client = GeminiClient(
        secure_1psid=psid,
        secure_1psidts=psidts or "",
        cookies=extra or None,
        proxy=args.proxy,
        account_index=args.account_index,
        verify=not args.skip_verify,
    )
    return client, json_cookies


async def _init_client(args: argparse.Namespace) -> tuple[GeminiClient, dict[str, str]]:
    if args.verbose:
        set_log_level("DEBUG")
    else:
        set_log_level("WARNING")

    client, json_cookies = _build_client(args)
    timeout = getattr(args, "request_timeout", 300)

    try:
        await client.init(
            timeout=timeout,
            auto_refresh=not args.no_auto_refresh,
            verbose=args.verbose,
        )
        return client, json_cookies
    except AuthError as e:
        raise SystemExit(
            f"Authentication failed: {e}\nPlease re-export cookies from your browser."
        ) from e


def _print_startup_diagnostics(client: GeminiClient) -> None:
    """Print account diagnostics and available models on startup."""
    print("=" * 60)
    print(" Gemini WebAPI - OpenAI Compatible Server")
    print("=" * 60)

    # 1. Account Diagnostics (from inspect)
    status = client.account_status
    print(f" Account Status: {status.name} ({status.value}) - {status.description}")

    abuse = client.abuse_status
    if abuse is None:
        print(" Abuse Status:   (not reported)")
    elif abuse.get("is_clean"):
        print(" Abuse Status:   clean")
    else:
        print(
            f" Abuse Status:   FLAGGED (code={abuse.get('status_code')}, signal={abuse.get('signal')})"
        )

    if usage := client.usage_info:
        tier = usage.get("tier", {})
        print(f" Plan Tier:      {tier.get('label') or '?'} ({tier.get('id')})")
        for key in ("current_5h", "weekly"):
            if metric := usage.get(key):
                reset_at = metric.get("reset_at")
                resets = f" (resets {reset_at})" if reset_at else ""
                print(
                    f"   - {metric.get('window', key):<8} {metric.get('usage_percentage')}% used, "
                    f"{metric.get('remaining_credits')} credits left{resets}"
                )
        if (credits := usage.get("ai_credits_remaining")) is not None:
            print(f"   - credits: {credits} AI credits remaining")

    quotas = client.quotas
    if quotas:
        print(" Quotas:")
        for quota_id, quota in quotas.items():
            if quota_id in ("extra", "usage_info"):
                continue
            lbl = quota.get("label", quota_id)
            rem = quota.get("remaining")
            tot = quota.get("total")
            pct = quota.get("usage_percentage")
            print(f"   - {lbl:<26} {rem}/{tot} remaining ({pct}% used)")
        if extra := quotas.get("extra", {}).get("default", {}):
            state = "BLOCKED" if extra.get("is_blocked") else "ok"
            pct = extra.get("usage_percentage")
            print(f"   - {'extra features':<26} {state} ({pct}% used)")

    # 2. Available Models (from models)
    models = client.list_models() or []
    print(f"\n Discovered Models ({len(models)}):")
    if models:
        name_w = max(len("Name"), max(len(m.model_name) for m in models))
        disp_w = max(len("Display"), max(len(m.display_name) for m in models))
        print(f"   {'Name':<{name_w}}  {'Display':<{disp_w}}  ID")
        print(f"   {'-' * (name_w + disp_w + 22)}")
        for m in models:
            mark = "" if m.is_available else "  (unavailable)"
            print(f"   {m.model_name:<{name_w}}  {m.display_name:<{disp_w}}  {m.model_id}{mark}")
    else:
        print("   (No models discovered)")

    print("=" * 60)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# region - OpenAI Format Helpers & Multimodal Parsing
# ---------------------------------------------------------------------------

DATA_URL_RE = re.compile(r"^data:(image/[a-zA-Z0-9\-\+\.]+);base64,(.+)$", re.DOTALL)


def _parse_messages(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Extract prompt text and file/image attachments from OpenAI messages."""
    prompt_parts: list[str] = []
    files: list[Any] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            if role == "system":
                prompt_parts.append(f"[System instruction: {content}]")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant: {content}]")
            else:
                prompt_parts.append(content)
        elif isinstance(content, list):
            # Multimodal content parts
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_val = part.get("text", "")
                    if role == "system":
                        prompt_parts.append(f"[System instruction: {text_val}]")
                    else:
                        prompt_parts.append(text_val)
                elif part_type == "image_url":
                    img_info = part.get("image_url")
                    url = img_info.get("url") if isinstance(img_info, dict) else str(img_info or "")
                    if not url:
                        continue
                    # Check for base64 data url
                    m = DATA_URL_RE.match(url)
                    if m:
                        mime_type, b64_data = m.groups()
                        try:
                            raw_bytes = base64.b64decode(b64_data)
                            bio = io.BytesIO(raw_bytes)
                            # Give it an extension based on mime type
                            ext = ".png"
                            if "jpeg" in mime_type or "jpg" in mime_type:
                                ext = ".jpg"
                            elif "webp" in mime_type:
                                ext = ".webp"
                            elif "gif" in mime_type:
                                ext = ".gif"
                            bio.name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
                            files.append(bio)
                        except Exception as e:
                            logger.warning(f"Failed to decode base64 image: {e}")
                    else:
                        # HTTP URL or local file path
                        files.append(url)

    # Prompt: combine extracted parts
    prompt = "\n\n".join(p for p in prompt_parts if p.strip()).strip()
    return prompt, files


def _format_images_to_markdown(output: Any) -> str:
    """Format any WebImage or GeneratedImage in ModelOutput as markdown."""
    if not output or not getattr(output, "images", None):
        return ""
    md_lines: list[str] = []
    web = [i for i in output.images if isinstance(i, WebImage)]
    gen = [i for i in output.images if isinstance(i, GeneratedImage)]

    if gen:
        md_lines.append("\n\n### Generated Images\n")
        for img in gen:
            md_lines.append(f"![Generated Image]({img.url})\n")
    if web:
        md_lines.append("\n\n### Referenced Images\n")
        for img in web:
            title = img.title or "Image"
            md_lines.append(f"![{title}]({img.url})\n")

    return "".join(md_lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 characters per token)."""
    return max(1, len(text) // 4) if text else 0


def _format_openai_model_list(client: GeminiClient) -> dict[str, Any]:
    models = client.list_models() or []
    created_time = int(time.time())
    data = []
    seen = set()
    for m in models:
        if m.model_name and m.model_name not in seen:
            seen.add(m.model_name)
            data.append(
                {
                    "id": m.model_name,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "google",
                    "permission": [],
                    "root": m.model_id,
                    "parent": None,
                }
            )
        if m.model_id and m.model_id not in seen:
            seen.add(m.model_id)
            data.append(
                {
                    "id": m.model_id,
                    "object": "model",
                    "created": created_time,
                    "owned_by": "google",
                    "permission": [],
                    "root": m.model_id,
                    "parent": None,
                }
            )
    return {"object": "list", "data": data}


def _format_openai_model_item(model_id: str, client: GeminiClient) -> dict[str, Any] | None:
    models = client.list_models() or []
    for m in models:
        if m.model_name == model_id or m.model_id == model_id:
            return {
                "id": m.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
                "permission": [],
                "root": m.model_id,
                "parent": None,
            }
    return None


SIMPLE_PROMPT_PREFIX = (
    "Be concise in your response. Answer only what has been asked. "
    "Avoid additional text, links or follow-up questions. "
    "Do not use any special markup unless explicitly stated in the question, "
    "by default use only simple ASCII characters."
)


# ---------------------------------------------------------------------------
# region - Server State & Handler
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self, client: GeminiClient, args: argparse.Namespace, json_cookies: dict[str, str]):
        self.client = client
        self.args = args
        self.json_cookies = json_cookies
        self.cookie_path = getattr(args, "cookies_json", None) or getattr(args, "cookies", None) or getattr(args, "cookes", None)
        self.chat_session: Any = None
        self.lock = asyncio.Lock()
        self.is_first_turn: bool = True
        self.is_running = True

    async def get_or_create_chat(self, model: str | None = None) -> Any:
        """Get or lazily initialize the shared ChatSession."""
        if self.chat_session is None:
            chosen_model = model or self.args.model
            gem = self.args.gem
            self.chat_session = self.client.start_chat(model=chosen_model, gem=gem)
            sys.stdout.write(
                f"[Server] Lazy-opened shared chat session (model: {chosen_model or 'default'})\n"
            )
            sys.stdout.flush()
        elif model and self.chat_session.model != model:
            self.chat_session.model = model
        return self.chat_session

    def reset_chat(self, model: str | None = None) -> Any:
        """Reset the shared chat session."""
        chosen_model = model or self.args.model
        gem = self.args.gem
        self.chat_session = self.client.start_chat(model=chosen_model, gem=gem)
        self.is_first_turn = True
        return self.chat_session

    async def cleanup(self) -> None:
        """Clean up on server shutdown."""
        self.is_running = False
        if self.cookie_path and not self.args.no_persist:
            _persist_cookies(
                self.cookie_path,
                self.json_cookies,
                self.client.cookies,
                verbose=self.args.verbose,
            )
        await self.client.close()


CORS_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Allow-Headers", "*"),
    ("Access-Control-Max-Age", "86400"),
]


def _log_success(client_ip: str, method: str, path: str, status: int, extra: str = "") -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{now_str}] {client_ip} \"{method} {path} HTTP/1.1\" {status} OK"
    if extra:
        msg += f" - {extra}"
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _log_error(client_ip: str, method: str, path: str, status: int, err: Exception) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"\n[{now_str}] ERROR {client_ip} \"{method} {path} HTTP/1.1\" {status}\n")
    sys.stderr.write(f"Exception: {type(err).__name__}: {err}\n")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.write("\n")
    sys.stderr.flush()


async def _send_json_response(
    writer: asyncio.StreamWriter,
    status_code: int,
    data: dict[str, Any],
    status_text: str = "OK",
) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = [
        f"HTTP/1.1 {status_code} {status_text}",
        "Content-Type: application/json; charset=utf-8",
        f"Content-Length: {len(body)}",
        "Connection: keep-alive",
    ]
    for k, v in CORS_HEADERS:
        headers.append(f"{k}: {v}")
    header_bytes = "\r\n".join(headers).encode("latin1") + b"\r\n\r\n"
    writer.write(header_bytes + body)
    await writer.drain()


async def _send_error_response(
    writer: asyncio.StreamWriter,
    status_code: int,
    message: str,
    err_type: str = "invalid_request_error",
    status_text: str = "Error",
) -> None:
    error_payload = {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": status_code,
        }
    }
    await _send_json_response(writer, status_code, error_payload, status_text=status_text)


# ---------------------------------------------------------------------------
# region - HTTP Request Handler
# ---------------------------------------------------------------------------


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    state: ServerState,
) -> None:
    peername = writer.get_extra_info("peername")
    client_ip = peername[0] if peername else "unknown"

    while state.is_running:
        try:
            # 1. Read request line
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            req_line = line_bytes.decode("utf-8", errors="replace").strip()
            if not req_line:
                continue

            parts = req_line.split()
            if len(parts) < 2:
                await _send_error_response(writer, 400, "Malformed request line", status_text="Bad Request")
                break
            method = parts[0].upper()
            raw_url = parts[1]
            http_version = parts[2] if len(parts) > 2 else "HTTP/1.1"

            # 2. Read headers
            headers: dict[str, str] = {}
            while True:
                header_line = await reader.readline()
                if not header_line or header_line in (b"\r\n", b"\n", b""):
                    break
                h_str = header_line.decode("utf-8", errors="replace").strip()
                if ":" in h_str:
                    h_k, h_v = h_str.split(":", 1)
                    headers[h_k.strip().lower()] = h_v.strip()

            # 3. Read body if Content-Length present
            body_bytes = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > 0:
                body_bytes = await reader.readexactly(content_length)

            # 4. Route request
            parsed_url = urlparse(raw_url)
            path = parsed_url.path.rstrip("/")
            if not path:
                path = "/"

            t0 = time.perf_counter()

            # Handle CORS OPTIONS
            if method == "OPTIONS":
                cors_response = (
                    b"HTTP/1.1 204 No Content\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: keep-alive\r\n"
                    + "\r\n".join(f"{k}: {v}" for k, v in CORS_HEADERS).encode("latin1")
                    + b"\r\n\r\n"
                )
                writer.write(cors_response)
                await writer.drain()
                continue

            # Route: GET /v1/models or /openai/v1/models or /models
            if method == "GET" and path in ("/v1/models", "/openai/v1/models", "/models"):
                data = _format_openai_model_list(state.client)
                await _send_json_response(writer, 200, data)
                dur = time.perf_counter() - t0
                _log_success(client_ip, method, path, 200, f"models_count={len(data['data'])} ({dur:.3f}s)")
                continue

            # Route: GET /v1/models/{model} or /openai/v1/models/{model}
            models_prefix = next(
                (p for p in ("/v1/models/", "/openai/v1/models/", "/models/") if path.startswith(p)),
                None,
            )
            if method == "GET" and models_prefix:
                model_id = path[len(models_prefix) :]
                item = _format_openai_model_item(model_id, state.client)
                if item:
                    await _send_json_response(writer, 200, item)
                    dur = time.perf_counter() - t0
                    _log_success(client_ip, method, path, 200, f"model={model_id} ({dur:.3f}s)")
                else:
                    await _send_error_response(writer, 404, f"Model '{model_id}' not found", status_text="Not Found")
                    _log_error(client_ip, method, path, 404, ValueError(f"Model '{model_id}' not found"))
                continue

            # Route: Health check / probe / account info
            if method == "GET" and path in ("/", "/health", "/healthz"):
                health_data = {
                    "status": "healthy",
                    "account_status": state.client.account_status.name,
                    "models_available": len(state.client.list_models() or []),
                    "temporary_chat": bool(state.args.temp),
                }
                await _send_json_response(writer, 200, health_data)
                _log_success(client_ip, method, path, 200)
                continue

            # Route: GET /v1/account or /openai/v1/account (diagnostics)
            if method == "GET" and path in ("/v1/account", "/openai/v1/account", "/account"):
                diag_data = {
                    "account_status": state.client.account_status.name,
                    "abuse_status": state.client.abuse_status,
                    "usage_info": state.client.usage_info,
                    "quotas": state.client.quotas,
                }
                await _send_json_response(writer, 200, diag_data)
                _log_success(client_ip, method, path, 200)
                continue

            # Route: POST /v1/chat/reset or /openai/v1/chat/reset
            if method == "POST" and path in ("/v1/chat/reset", "/openai/v1/chat/reset", "/chat/reset"):
                async with state.lock:
                    state.reset_chat()
                await _send_json_response(writer, 200, {"status": "ok", "message": "Shared chat session reset"})
                _log_success(client_ip, method, path, 200, "shared session reset")
                continue

            # Route: POST /v1/chat/completions or /openai/v1/chat/completions or /chat/completions
            if method == "POST" and path in (
                "/v1/chat/completions",
                "/openai/v1/chat/completions",
                "/chat/completions",
            ):
                try:
                    payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                except Exception as e:
                    await _send_error_response(writer, 400, f"Invalid JSON body: {e}", status_text="Bad Request")
                    _log_error(client_ip, method, path, 400, e)
                    continue

                messages = payload.get("messages", [])
                if not messages or not isinstance(messages, list):
                    await _send_error_response(
                        writer, 400, "'messages' is required and must be a list", status_text="Bad Request"
                    )
                    _log_error(client_ip, method, path, 400, ValueError("Missing or invalid 'messages'"))
                    continue

                requested_model = payload.get("model") or state.args.model
                is_stream = bool(payload.get("stream", False))
                prompt, files = _parse_messages(messages)

                if not prompt and not files:
                    await _send_error_response(writer, 400, "Empty prompt in messages", status_text="Bad Request")
                    _log_error(client_ip, method, path, 400, ValueError("Empty prompt"))
                    continue

                # Serialize access to the shared chat session
                async with state.lock:
                    if state.args.simple and state.is_first_turn:
                        prompt = f"{SIMPLE_PROMPT_PREFIX}\n\n{prompt}"
                        state.is_first_turn = False

                    chat = await state.get_or_create_chat(model=requested_model)
                    chat_model_name = getattr(chat, "model", None) or requested_model or "gemini"

                    if is_stream:
                        # ---------------- STREAMING RESPONSE ----------------
                        headers_sse = [
                            "HTTP/1.1 200 OK",
                            "Content-Type: text/event-stream; charset=utf-8",
                            "Cache-Control: no-cache, no-transform",
                            "Connection: keep-alive",
                            "Transfer-Encoding: chunked",
                        ]
                        for k, v in CORS_HEADERS:
                            headers_sse.append(f"{k}: {v}")
                        header_bytes = "\r\n".join(headers_sse).encode("latin1") + b"\r\n\r\n"
                        writer.write(header_bytes)
                        await writer.drain()

                        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                        created_ts = int(time.time())
                        prompt_tokens = _estimate_tokens(prompt)
                        completion_chars = 0
                        last_output = None

                        async def _send_sse_chunk(chunk_dict: dict[str, Any]) -> None:
                            chunk_str = f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"
                            chunk_data = chunk_str.encode("utf-8")
                            chunk_header = f"{len(chunk_data):X}\r\n".encode("latin1")
                            writer.write(chunk_header + chunk_data + b"\r\n")
                            await writer.drain()

                        # Stream start (role)
                        await _send_sse_chunk(
                            {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": chat_model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": ""},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                        try:
                            async for output_chunk in chat.send_message_stream(
                                prompt=prompt,
                                files=files or None,
                                temporary=bool(state.args.temp),
                            ):
                                last_output = output_chunk
                                # Thoughts delta (reasoning content)
                                if getattr(output_chunk, "thoughts_delta", None):
                                    await _send_sse_chunk(
                                        {
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_ts,
                                            "model": chat_model_name,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "reasoning_content": output_chunk.thoughts_delta
                                                    },
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )

                                # Text delta
                                if getattr(output_chunk, "text_delta", None):
                                    delta_text = output_chunk.text_delta
                                    completion_chars += len(delta_text)
                                    await _send_sse_chunk(
                                        {
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_ts,
                                            "model": chat_model_name,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": delta_text},
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )

                            # Format any images from last_output
                            img_md = _format_images_to_markdown(last_output)
                            if img_md:
                                completion_chars += len(img_md)
                                await _send_sse_chunk(
                                    {
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": chat_model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": img_md},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )

                            # Final chunk with finish_reason and usage
                            comp_tokens = _estimate_tokens("x" * completion_chars)
                            await _send_sse_chunk(
                                {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": chat_model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop",
                                        }
                                    ],
                                    "usage": {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": comp_tokens,
                                        "total_tokens": prompt_tokens + comp_tokens,
                                    },
                                }
                            )

                            # Send [DONE]
                            done_payload = b"data: [DONE]\n\n"
                            writer.write(f"{len(done_payload):X}\r\n".encode("latin1") + done_payload + b"\r\n")
                            # Terminate chunked transfer
                            writer.write(b"0\r\n\r\n")
                            await writer.drain()

                            dur = time.perf_counter() - t0
                            _log_success(
                                client_ip,
                                method,
                                path,
                                200,
                                f"stream=True, model={chat_model_name}, tokens={prompt_tokens}/{comp_tokens} ({dur:.2f}s)",
                            )
                        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
                            # Client disconnected early
                            break
                        except Exception as e:
                            _log_error(client_ip, method, path, 500, e)
                            # Cannot send headers again, just end chunked stream
                            writer.write(b"0\r\n\r\n")
                            await writer.drain()
                        continue

                    else:
                        # ---------------- NON-STREAMING RESPONSE ----------------
                        try:
                            output = await chat.send_message(
                                prompt=prompt,
                                files=files or None,
                                temporary=bool(state.args.temp),
                            )
                            full_text = (output.text or "") + _format_images_to_markdown(output)
                            prompt_tokens = _estimate_tokens(prompt)
                            comp_tokens = _estimate_tokens(full_text)

                            msg_obj: dict[str, Any] = {
                                "role": "assistant",
                                "content": full_text,
                            }
                            if getattr(output, "thoughts", None):
                                msg_obj["reasoning_content"] = output.thoughts

                            response_payload = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                                "object": "chat.completion",
                                "created": int(time.time()),
                                "model": chat_model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": msg_obj,
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": comp_tokens,
                                    "total_tokens": prompt_tokens + comp_tokens,
                                },
                            }
                            await _send_json_response(writer, 200, response_payload)
                            dur = time.perf_counter() - t0
                            _log_success(
                                client_ip,
                                method,
                                path,
                                200,
                                f"model={chat_model_name}, tokens={prompt_tokens}/{comp_tokens} ({dur:.2f}s)",
                            )
                        except Exception as e:
                            await _send_error_response(writer, 500, f"Gemini generation error: {e}", status_text="Internal Server Error")
                            _log_error(client_ip, method, path, 500, e)
                        continue

            # Fallback: 404
            await _send_error_response(writer, 404, f"Path '{path}' not found", status_text="Not Found")
            _log_error(client_ip, method, path, 404, ValueError(f"Unknown path '{path}'"))

        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            break
        except Exception as e:
            _log_error(client_ip, "UNKNOWN", "UNKNOWN", 500, e)
            break

    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# region - Argparse & Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server backed by gemini-webapi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Server network options
    parser.add_argument(
        "--addr",
        "--host",
        default="127.0.0.1",
        help="IP address / hostname to listen on",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=4981,
        help="Port number to listen on",
    )

    # Session & Chat options
    parser.add_argument(
        "--temp",
        "--temporary",
        action="store_true",
        help="Use temporary chat session (not saved to account history)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Prepend concise plain ASCII instructions to the first prompt of the conversation",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Default model name, alias, or ID (e.g. gemini-flash, gemini-pro)",
    )
    parser.add_argument(
        "--gem",
        default=None,
        help="Gem ID or name to use as system prompt / personality",
    )

    # Authentication & Cookies
    parser.add_argument(
        "--cookies",
        "--cookies-json",
        "--cookes",
        dest="cookies_json",
        default=None,
        help="Path to JSON cookies file exported from browser",
    )
    parser.add_argument(
        "--account-index",
        type=int,
        default=None,
        help="Google account index (for multi-account sessions)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not write updated session cookies back to disk",
    )
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable automatic cookie background refresh",
    )

    # Network / Connection options
    parser.add_argument(
        "--proxy",
        default=(
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
        ),
        help="HTTP/HTTPS proxy URL",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300,
        help="Per-request Gemini timeout in seconds",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SSL certificate verification",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


async def run_server(args: argparse.Namespace) -> None:
    client, json_cookies = await _init_client(args)

    # Print startup diagnostics (stats & models)
    _print_startup_diagnostics(client)

    state = ServerState(client, args, json_cookies)

    # Create standard asyncio TCP server
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, state),
        host=args.addr,
        port=args.port,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    print(f"[Server] Listening on {addrs}")
    print(f"[Server] Temporary chat: {bool(args.temp)}")
    print(f"[Server] Simple mode:    {bool(args.simple)}")
    print(f"[Server] Endpoint ready: http://{args.addr}:{args.port}/openai/v1/chat/completions")
    print(f"[Server] (Press Ctrl+C to stop)\n")
    sys.stdout.flush()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal() -> None:
        sys.stdout.write("\n[Server] Shutdown signal received...\n")
        sys.stdout.flush()
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _on_signal)

    try:
        await stop_event.wait()
    finally:
        server.close()
        await server.wait_closed()
        await state.cleanup()
        print("[Server] Server closed cleanly.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args))
        return 0
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit):
            return int(e.code if isinstance(e.code, int) else 1)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
