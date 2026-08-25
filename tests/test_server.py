"""Unit tests for server.py OpenAI-compatible API server."""

import asyncio
import base64
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Make repo root and src importable
if (_root := str(Path(__file__).resolve().parent.parent)) not in sys.path:
    sys.path.insert(0, _root)
if (_src := str(Path(__file__).resolve().parent.parent / "src")) not in sys.path:
    sys.path.insert(0, _src)

from server import (
    ServerState,
    _estimate_tokens,
    _format_images_to_markdown,
    _format_openai_model_item,
    _format_openai_model_list,
    _load_cookies_with_meta,
    _parse_expiry,
    _parse_messages,
    _persist_cookies,
    build_parser,
    handle_connection,
)
from gemini_webapi.constants import AccountStatus
from gemini_webapi.types.availablemodel import AvailableModel
from gemini_webapi.types.candidate import Candidate
from gemini_webapi.types.image import GeneratedImage, WebImage
from gemini_webapi.types.modeloutput import ModelOutput


class TestServerHelpers(unittest.TestCase):
    def test_parse_expiry(self):
        self.assertIsNone(_parse_expiry(None))
        self.assertEqual(_parse_expiry(1700000000), 1700000000)
        self.assertEqual(_parse_expiry("1700000000"), 1700000000)
        self.assertIsInstance(_parse_expiry("2026-08-24T20:00:00Z"), int)

    def test_load_cookies_flat_dict(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write(json.dumps({"__Secure-1PSID": "psid_val", "__Secure-1PSIDTS": "ts_val"}))
            f.flush()
            tmp_path = f.name

        try:
            cookies, meta = _load_cookies_with_meta(tmp_path)
            self.assertEqual(cookies["__Secure-1PSID"], "psid_val")
            self.assertEqual(cookies["__Secure-1PSIDTS"], "ts_val")
        finally:
            os.unlink(tmp_path)

    def test_load_cookies_nested_list(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write(
                json.dumps(
                    [
                        {"name": "__Secure-1PSID", "value": "psid_val", "expirationDate": 1800000000},
                        {"name": "__Secure-1PSIDTS", "value": "ts_val"},
                    ]
                )
            )
            f.flush()
            tmp_path = f.name

        try:
            cookies, meta = _load_cookies_with_meta(tmp_path)
            self.assertEqual(cookies["__Secure-1PSID"], "psid_val")
            self.assertEqual(meta["__Secure-1PSID"]["expires_epoch"], 1800000000)
        finally:
            os.unlink(tmp_path)

    def test_parse_messages_plain(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        prompt, files = _parse_messages(messages)
        self.assertIn("[System instruction: You are a helpful assistant.]", prompt)
        self.assertIn("Hello!", prompt)
        self.assertEqual(files, [])

    def test_parse_messages_multimodal_base64(self):
        sample_b64 = base64.b64encode(b"fake_image_bytes").decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample_b64}"},
                    },
                ],
            }
        ]
        prompt, files = _parse_messages(messages)
        self.assertEqual(prompt, "Describe this image")
        self.assertEqual(len(files), 1)
        self.assertIsInstance(files[0], io.BytesIO)
        self.assertEqual(files[0].getvalue(), b"fake_image_bytes")

    def test_token_estimation(self):
        self.assertEqual(_estimate_tokens(""), 0)
        self.assertEqual(_estimate_tokens("Hello world"), 2)

    def test_format_images_to_markdown(self):
        output = MagicMock()
        web_img = WebImage(url="http://example.com/img.png", title="Example")
        gen_img = GeneratedImage(url="http://example.com/gen.png")
        output.images = [web_img, gen_img]
        md = _format_images_to_markdown(output)
        self.assertIn("![Example](http://example.com/img.png)", md)
        self.assertIn("![Generated Image](http://example.com/gen.png)", md)


class TestServerCLIParser(unittest.TestCase):
    def test_cli_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.port, 4981)
        self.assertEqual(args.addr, "127.0.0.1")
        self.assertFalse(args.temp)
        self.assertIsNone(args.cookies_json)

    def test_cli_parser_custom_options(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--port",
                "8080",
                "--addr",
                "0.0.0.0",
                "--temp",
                "--simple",
                "--cookies",
                "/tmp/cookies.json",
                "--model",
                "gemini-2.5-pro",
                "--gem",
                "gem-123",
                "--no-persist",
                "--verbose",
            ]
        )
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.addr, "0.0.0.0")
        self.assertTrue(args.temp)
        self.assertTrue(args.simple)
        self.assertEqual(args.cookies_json, "/tmp/cookies.json")
        self.assertEqual(args.model, "gemini-2.5-pro")
        self.assertEqual(args.gem, "gem-123")
        self.assertTrue(args.no_persist)
        self.assertTrue(args.verbose)


class TestServerIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Build mock GeminiClient
        self.mock_client = MagicMock()
        self.mock_client.account_status = AccountStatus.AVAILABLE
        self.mock_client.abuse_status = {"is_clean": True}
        self.mock_client.usage_info = {
            "tier": {"id": "1", "label": "Gemini Advanced"},
            "current_5h": {"window": "5h", "usage_percentage": 5, "remaining_credits": 95},
        }
        self.mock_client.quotas = {
            "flash": {"label": "Gemini Flash", "remaining": 100, "total": 100, "usage_percentage": 0}
        }
        mock_model_1 = AvailableModel(
            model_name="gemini-flash",
            display_name="Gemini Flash",
            model_id="models/gemini-flash",
            description="Fast model",
            capacity=1,
            is_available=True,
        )
        mock_model_2 = AvailableModel(
            model_name="gemini-pro",
            display_name="Gemini Pro",
            model_id="models/gemini-pro",
            description="Pro model",
            capacity=1,
            is_available=True,
        )
        self.mock_client.list_models.return_value = [mock_model_1, mock_model_2]
        self.mock_client.close = AsyncMock()

        # Build mock ChatSession
        self.mock_chat = MagicMock()
        self.mock_chat.model = "gemini-flash"

        async def fake_send_message(prompt, files=None, temporary=False, **kwargs):
            return ModelOutput(
                metadata=["c_test", "r_test"],
                candidates=[
                    Candidate(
                        rcid="rc_test",
                        text=f"Response to: {prompt}",
                        text_delta=f"Response to: {prompt}",
                        thoughts="Mock thinking process",
                    )
                ],
            )

        async def fake_send_message_stream(prompt, files=None, temporary=False, **kwargs):
            yield ModelOutput(
                metadata=["c_test", "r_test"],
                candidates=[
                    Candidate(
                        rcid="rc_test",
                        text="Thinking...",
                        thoughts_delta="Thinking...",
                    )
                ],
            )
            yield ModelOutput(
                metadata=["c_test", "r_test"],
                candidates=[
                    Candidate(
                        rcid="rc_test",
                        text="Hello ",
                        text_delta="Hello ",
                    )
                ],
            )
            yield ModelOutput(
                metadata=["c_test", "r_test"],
                candidates=[
                    Candidate(
                        rcid="rc_test",
                        text="world!",
                        text_delta="world!",
                    )
                ],
            )

        self.mock_chat.send_message = AsyncMock(side_effect=fake_send_message)
        self.mock_chat.send_message_stream = fake_send_message_stream
        self.mock_client.start_chat.return_value = self.mock_chat

        # Server args & state
        parser = build_parser()
        self.args = parser.parse_args(["--port", "0", "--addr", "127.0.0.1"])
        self.state = ServerState(self.mock_client, self.args, {})

        # Start server on ephemeral port
        self.server = await asyncio.start_server(
            lambda r, w: handle_connection(r, w, self.state),
            host="127.0.0.1",
            port=0,
        )
        self.port = self.server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        await self.state.cleanup()

    def _http_request(self, path: str, method: str = "GET", data: dict | None = None, headers: dict | None = None) -> tuple[int, dict | str, dict]:
        url = f"{self.base_url}{path}"
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        body = json.dumps(data).encode("utf-8") if data is not None else None

        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                resp_bytes = response.read()
                resp_headers = dict(response.headers)
                try:
                    resp_data = json.loads(resp_bytes.decode("utf-8"))
                except Exception:
                    resp_data = resp_bytes.decode("utf-8")
                return response.status, resp_data, resp_headers
        except urllib.error.HTTPError as e:
            err_bytes = e.read()
            try:
                err_data = json.loads(err_bytes.decode("utf-8"))
            except Exception:
                err_data = err_bytes.decode("utf-8")
            return e.code, err_data, dict(e.headers)

    async def test_health_check(self):
        loop = asyncio.get_running_loop()
        status, data, headers = await loop.run_in_executor(None, self._http_request, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["account_status"], "AVAILABLE")

    async def test_get_models(self):
        loop = asyncio.get_running_loop()
        status, data, headers = await loop.run_in_executor(None, self._http_request, "/openai/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "list")
        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("gemini-flash", model_ids)
        self.assertIn("gemini-pro", model_ids)

    async def test_get_single_model(self):
        loop = asyncio.get_running_loop()
        status, data, headers = await loop.run_in_executor(None, self._http_request, "/v1/models/gemini-flash")
        self.assertEqual(status, 200)
        self.assertEqual(data["id"], "gemini-flash")
        self.assertEqual(data["object"], "model")

    async def test_chat_completions_non_stream(self):
        loop = asyncio.get_running_loop()
        payload = {
            "model": "gemini-flash",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
        status, data, headers = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/completions", method="POST", data=payload),
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "gemini-flash")
        self.assertEqual(len(data["choices"]), 1)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(data["choices"][0]["message"]["content"], "Response to: Hello!")
        self.assertEqual(data["choices"][0]["message"]["reasoning_content"], "Mock thinking process")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", data)

    async def test_chat_completions_streaming(self):
        loop = asyncio.get_running_loop()

        def _fetch_stream():
            url = f"{self.base_url}/openai/v1/chat/completions"
            payload = json.dumps(
                {
                    "model": "gemini-flash",
                    "messages": [{"role": "user", "content": "Hello streaming!"}],
                    "stream": True,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            chunks = []
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertIn("text/event-stream", resp.headers.get("Content-Type", ""))
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        chunks.append(decoded)
            return chunks

        lines = await loop.run_in_executor(None, _fetch_stream)
        data_lines = [l for l in lines if l.startswith("data:")]
        self.assertTrue(len(data_lines) >= 3)
        self.assertIn("data: [DONE]", data_lines)

        # Parse first JSON chunk (role)
        first_json = json.loads(data_lines[0].replace("data: ", ""))
        self.assertEqual(first_json["object"], "chat.completion.chunk")
        self.assertEqual(first_json["choices"][0]["delta"]["role"], "assistant")

    async def test_reset_session(self):
        loop = asyncio.get_running_loop()
        status, data, headers = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/reset", method="POST"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    async def test_cors_options(self):
        loop = asyncio.get_running_loop()
        status, data, headers = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/completions", method="OPTIONS"),
        )
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")

    async def test_error_handling_malformed_body(self):
        loop = asyncio.get_running_loop()
        url = f"{self.base_url}/openai/v1/chat/completions"
        req = urllib.request.Request(
            url, data=b"{invalid_json", headers={"Content-Type": "application/json"}, method="POST"
        )
        def _send():
            try:
                urllib.request.urlopen(req, timeout=5)
                return 200, None
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode("utf-8"))

        status, data = await loop.run_in_executor(None, _send)
        self.assertEqual(status, 400)
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], 400)

    async def test_simple_mode_prepends_first_turn_only(self):
        loop = asyncio.get_running_loop()
        # Enable simple mode on server state
        self.state.args.simple = True
        self.state.is_first_turn = True

        # First turn
        payload_1 = {
            "model": "gemini-flash",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        }
        status_1, data_1, _ = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/completions", method="POST", data=payload_1),
        )
        self.assertEqual(status_1, 200)
        # Verify first call included the simple prefix
        first_call_args = self.mock_chat.send_message.call_args_list[0]
        prompt_sent_1 = first_call_args.kwargs.get("prompt") or first_call_args.args[0]
        self.assertIn("Be concise in your response", prompt_sent_1)
        self.assertIn("What is 2+2?", prompt_sent_1)

        # Second turn (should NOT prepend prefix again)
        payload_2 = {
            "model": "gemini-flash",
            "messages": [{"role": "user", "content": "And 3+3?"}],
        }
        status_2, data_2, _ = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/completions", method="POST", data=payload_2),
        )
        self.assertEqual(status_2, 200)
        second_call_args = self.mock_chat.send_message.call_args_list[1]
        prompt_sent_2 = second_call_args.kwargs.get("prompt") or second_call_args.args[0]
        self.assertNotIn("Be concise in your response", prompt_sent_2)
        self.assertEqual(prompt_sent_2, "And 3+3?")

        # Reset session -> Next turn should prepend prefix again
        await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/reset", method="POST"),
        )
        payload_3 = {
            "model": "gemini-flash",
            "messages": [{"role": "user", "content": "New question?"}],
        }
        status_3, data_3, _ = await loop.run_in_executor(
            None,
            lambda: self._http_request("/openai/v1/chat/completions", method="POST", data=payload_3),
        )
        self.assertEqual(status_3, 200)
        third_call_args = self.mock_chat.send_message.call_args_list[2]
        prompt_sent_3 = third_call_args.kwargs.get("prompt") or third_call_args.args[0]
        self.assertIn("Be concise in your response", prompt_sent_3)
        self.assertIn("New question?", prompt_sent_3)


if __name__ == "__main__":
    unittest.main()
