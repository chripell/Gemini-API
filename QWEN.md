# QWEN.md

Instructions for working in this repository.

## Project overview

**Gemini-API** (`gemini-webapi`) is a reverse-engineered, fully **asynchronous** Python wrapper for the
[Google Gemini](https://gemini.google.com) web app (formerly Bard). It talks to the web app's internal
`batchexecute` / gRPC-style endpoints using browser cookies for auth — there is **no official API key** and
**no HTTP framework dependency**.

Three deliverables live in this repo:

| Entry point | What it is |
| :-- | :-- |
| `src/gemini_webapi/` | The published PyPI package (`gemini-webapi`) — the library API |
| `cli.py` | Standalone terminal client (single-shot ask, chat, deep research, downloads, diagnostics) |
| `server.py` | OpenAI-compatible HTTP server backed by the library (stdlib `asyncio` only, no FastAPI/uvicorn) |

Key capabilities: text/thought streaming, file upload input, image/video/audio generation, Gems as system
prompts, Gemini extensions, multi-candidate replies, chat history read/delete, deep research, background
cookie auto-refresh, and optional browser cookie import (`browser-cookie3`).

- **Language / runtime:** Python **3.11+** (`requires-python = ">=3.11"`, ruff `target-version = "py311"`)
- **Core deps:** `curl-cffi` (browser-impersonating async HTTP), `pydantic` v2 (all response/DTO types),
  `orjson` (JSON), `loguru` (logging)
- **Versioning:** dynamic via `setuptools_scm` (git tags `vX.Y.Z`) — never hardcode a version in source
- **License:** see `LICENSE`; package author/maintainer info in `pyproject.toml`

## Repository layout

```
.
├── cli.py                     # CLI tool (argparse subcommands)
├── server.py                  # OpenAI-compatible API server
├── README.md                  # Library usage reference (authoritative for public API)
├── server.md                  # server.py endpoint/option reference
├── pyproject.toml             # packaging + ruff + pyright config
├── src/gemini_webapi/
│   ├── client.py              # GeminiClient (composes the mixins) + ChatSession  (~2.6k lines)
│   ├── constants.py           # Endpoint, GRPC, Headers, AccountStatus, ErrorCode, Model (deprecated)
│   ├── exceptions.py          # AuthError, APIError, GeminiError, TimeoutError, UsageLimitExceededError, ...
│   ├── components/            # Feature mixins: chat_mixin, gem_mixin, research_mixin
│   ├── types/                 # pydantic models: ModelOutput, Candidate, Image, Video, Gem, AvailableModel, ...
│   └── utils/                 # parsing, upload, cookie rotation, access token, citations, logger, decorators
└── tests/                     # unittest suite (see Testing below)
```

Architecture notes:

- **`GeminiClient(ChatMixin, GemMixin, ResearchMixin)`** — features are split into mixins, and each mixin
  declares the members it expects from the composed client inside an `if TYPE_CHECKING:` block. When adding a
  cross-component dependency, add the protocol-style declaration there rather than importing `client` at runtime
  (that would create a cycle).
- **All network calls funnel through** `GeminiClient._batch_execute` and the `GRPC` endpoint enum in
  `constants.py`; responses are RPC-enveloped strings unwrapped by `_parse_rpc_results` plus the helpers in
  `utils/parsing.py` (`extract_json_from_response`, `get_nested_value`, `StreamingFrameParser`, ...).
- **`ModelOutput`** is a thin facade over `candidates[chosen]` — new response fields are usually added to
  `types/candidate.py` and exposed as a `@property` on `ModelOutput`.
- **Public surface** is re-exported from `src/gemini_webapi/__init__.py` (`from .types import *`,
  `from .exceptions import *`). New user-facing types belong in that chain.
- **`server.py`/`cli.py` are root scripts, not package modules.** Both prepend `src/` to `sys.path` so they run
  from a checkout without installation. `server.py` is organized with `# region - ...` markers.

## Building and running

```sh
# Install the package in editable mode (with the optional browser cookie import extra)
pip install -e ".[browser]"        # or: pip install -U gemini_webapi

# Dev tooling extra (pyright, ruff, ty); pyproject also declares a PEP 735 `dev` dependency group for uv
pip install -e ".[dev]"

# Run the library straight from a source checkout without installing
PYTHONPATH=src python -c "import gemini_webapi; print(gemini_webapi.__version__ if hasattr(gemini_webapi,'__version__') else 'ok')"

# Build a distribution (what CI does before publishing)
python -m pip install build && python -m build
```

### CLI tool

```sh
python cli.py --cookies-json cookies.json ask "What is quantum computing?"     # streams by default
python cli.py --cookies-json cookies.json ask --no-stream --image photo.jpg "Describe this"
python cli.py --cookies-json cookies.json reply c_abc123 "Tell me more"
python cli.py --cookies-json cookies.json list                    # recent chats
python cli.py --cookies-json cookies.json read c_abc123           # one conversation
python cli.py --cookies-json cookies.json models                  # models this account can use
python cli.py --cookies-json cookies.json inspect                 # status, quotas, usage limits
python cli.py --cookies-json cookies.json research send --prompt "AI chip competition 2025"
python cli.py --cookies-json cookies.json research check c_abc123 --research-id 0f3d...
python cli.py --cookies-json cookies.json research get c_abc123 --output report.md
```

Global options go **before** the subcommand: `--cookies-json`, `--proxy`, `--model`, `--verbose`,
`--no-persist`, `--request-timeout`. The CLI rewrites rotated cookies back into the JSON file unless
`--no-persist` is given.

### OpenAI-compatible server

```sh
python server.py --cookies cookies.json --port 4981        # binds 127.0.0.1 by default
python server.py --help                                    # full option list
```

Endpoints (each also served under the `/openai` prefix): `POST /v1/chat/completions` (JSON + SSE streaming,
reasoning exposed as `delta.reasoning_content`), `GET /v1/models`, `GET /v1/models/{model}`,
`POST /v1/chat/reset`, `GET /v1/account`, `GET /health`. Cookies may come from env instead of a file:
`GEMINI_SECURE_1PSID` / `GEMINI_SECURE_1PSIDTS` (also `SECURE_1PSID` / `SECURE_1PSIDTS`).
See `server.md` for request/response examples; `--temp`, `--simple`, `--gem`, `--model` change conversation behavior.

### Authentication / cookies

- Auth needs `__Secure-1PSID` (plus `__Secure-1PSIDTS` when present) copied from an authenticated
  `gemini.google.com` session, or `browser-cookie3` (Firefox is the reliable choice; Chromium browsers use
  Device Bound Session Credentials and expire within hours).
- Background auto-refresh is **on by default** (`client.init(auto_refresh=True)`). Rotated cookies are cached at
  `GEMINI_COOKIE_PATH` if set, else `<tempdir>/gemini_webapi/.cached_cookies_<1PSID>.json`.
- Never commit cookie files or secrets — `.gitignore` already covers `cookies.json`, `*.cookies`, `.env`,
  `temp/`, `.temp/`.

## Testing

The suite uses **`unittest`** (`.vscode/settings.json` explicitly disables pytest and enables unittest
discovery on `./tests` for `test_*.py`).

```sh
# Everything
PYTHONPATH=src python3 -m unittest discover -s tests

# A single module / class / method
PYTHONPATH=src python3 -m unittest tests.test_server.TestServerHelpers -v
PYTHONPATH=src python3 -m unittest tests.test_cli.TestCLITool

# Offline-only subset (fast, no network)
python3 -m unittest tests.test_server.TestServerHelpers tests.test_server.TestServerCLIParser tests.test_cli
```

- **Always set `PYTHONPATH=src`** (or `pip install -e .`) when running a single test module. Only
  `test_cli.py` and `test_server.py` inject the repo root and `src/` into `sys.path` themselves; the
  library-side modules import `gemini_webapi` directly and fail with `ModuleNotFoundError` on their own.
  `discover -s tests` happens to work without it only because `test_cli.py` sorts first and its path
  insertion is a global side effect — do not rely on that.
- `test_client_features.py`, `test_gem_mixin.py`, `test_save_image.py`, `test_deep_research.py` are **live
  integration tests** — they make real requests against `gemini.google.com` and consume account quota. They read
  `SECURE_1PSID` / `SECURE_1PSIDTS` from the environment and call `self.skipTest(...)` on `AuthError` or when
  `account_status != AccountStatus.AVAILABLE` (guest fallback).
  **On this machine `browser-cookie3` is installed and a logged-in browser session exists, so they do not skip —
  they run and take tens of seconds per class.** Confirm before running them repeatedly.
- `test_server.py` and `test_cli.py` are the unit-level ones: `unittest.mock` (`AsyncMock`, `patch`) plus a
  locally launched server socket. Prefer adding mock-based coverage here for new `server.py`/`cli.py` logic.
- Async tests use `unittest.IsolatedAsyncioTestCase` with `asyncSetUp` / `asyncTearDown`
  (`await client.close()` in teardown) — follow that pattern instead of hand-rolling event loops.
- Run the offline subset and a lint pass before declaring work complete; state clearly if live tests were
  skipped rather than passing.

## Development conventions

**Style & tooling** (configured in `pyproject.toml`; install with `pip install -e ".[dev]"`):

```sh
ruff check src tests cli.py server.py      # lint
ruff format src tests cli.py server.py     # format
pyright src                                # type check
```

- `ruff`: line-length **100**, double quotes, space indent; `E501` is ignored because the formatter enforces
  length. Enabled rule families include `B`, `C4`, `E`/`W`, `F`, `G`, `I` (isort), `LOG`, `N`, `PIE`, `PT`,
  `RET`, `RUF`, `SIM`, `TID`, `UP`. Expect `UP`/`I`/`N`/`RET`/`SIM` suggestions on new code — use
  `collections.abc` types, sorted imports, and PEP 8 naming.
- `pyright`: `typeCheckingMode = "standard"`. Annotate public functions; the codebase uses modern syntax
  (`str | None`, `dict[str, Any]`, `list[...]`) with **no** `from __future__ import annotations`.
- Code style badge in `README.md` is Black; the enforced formatter is `ruff format` (Black-compatible).

**Codebase idioms to mirror:**

- Imports inside the package are **absolute** (`from gemini_webapi.constants import GRPC`), not relative.
- Data shapes are **pydantic v2 `BaseModel`** classes in `types/` (`ConfigDict` where extra strictness is
  needed); `RPCData` wraps gRPC payloads. Use `@validate_call` where a public helper needs runtime coercion
  (see `utils/upload_file.py`).
- Enumerations use `IntEnum` / `StrEnum` (`AccountStatus`, `ErrorCode`, `Endpoint`, `GRPC`, `Field`).
- `GeminiClient` uses `__slots__` — **add any new client attribute to `__slots__`** or assignment fails.
- `import orjson as json` is the package convention for JSON handling (`orjson` returns bytes; existing code
  decodes accordingly). Logging goes through the shared `logger` / `set_log_level` in `utils/logger.py`
  (loguru) — first `set_log_level` call removes all existing loguru handlers globally.
- The `@running(retry=N)` decorator (`utils/decorators.py`) guards the request core (`_generate`,
  `_batch_execute`); it supports coroutines and async generators and re-runs the call on `APIError`. Public
  methods reached through it do not need their own liveness check.
- Docstrings are **NumPy style** with a `Parameters` section, backtick-quoted types
  (`` secure_1psid: `str`, optional ``) and a `Raises` section where relevant.
- Comments are sparse and explain non-obvious web-app reverse-engineering constraints (flag indices, watchdog
  retries, cookie race handling) — do not add narrative comments.
- Errors: raise the typed exceptions from `exceptions.py` rather than generic `RuntimeError`. Note that
  `exceptions.TimeoutError` (a `GeminiError` subclass) has no `__all__` guard and is re-exported from the package
  root, so `from gemini_webapi import *` shadows the builtin `TimeoutError` in the importing module.

**Model selection (important, recently changed):** model discovery is **dynamic** at `init()` time. Use
`client.list_models()` / `client.resolve_model(name)` and accept names, aliases, display names, ids, or an
`AvailableModel`. The `Model` enum in `constants.py` is **deprecated** (its `PLUS_`/`ADVANCED_` variants encode
account tiers the client now reads from the account); `Model.from_name` / `Model.from_dict` have already been
removed. Passing an enum member still works but logs a warning via `warn_deprecated_model` — do not introduce
new code that depends on it.

**Deep research** is modeled as two turns of one chat session (plan → confirm) with server-side execution; the
finished report arrives as an inline document (`output.deep_research_document`), not as reply text. Keep that
shape when touching `components/research_mixin.py`.

## Useful facts / gotchas

- Everything is async; library examples and docs use `asyncio.run(main())`. There is no sync facade.
- Multi-turn state lives in `ChatSession.metadata`: a 10-slot list seeded from `DEFAULT_METADATA`
  (`constants.py`) where `cid`/`rid`/`rcid` are indices 0/1/2; the setter only overwrites non-`None` entries, so
  partial metadata can be applied safely. Persist that list to resume conversations across processes.
- Temporary chats (`temporary=True`) are not saved to Gemini history — use them in tests that would otherwise
  litter the account.
- Quota/account diagnostics are populated during init and exposed as `client.quotas`, `client.usage_info`,
  `client.abuse_status`, `client.account_status`; `cli.py inspect` prints them.
- Region/plan gated features (image/video generation, extensions, deep research) may be unavailable on the
  test account — code paths degrade by returning empty lists, and tests skip rather than assert availability.
- GitHub Actions only builds/publishes on `vX.Y.Z` tags (`.github/workflows/pypi-publish.yml`,
  `github-release.yml`); there is **no** CI lint or test workflow, so run ruff/pyright/unittest locally.
- This working copy tracks upstream as `upstream/master` and publishes from a fork remote; feature branches
  (e.g. `chri`) are normal here. Commit messages are short imperative sentences without conventional-commit
  prefixes (`Add OpenAI-compatible API server`, `Resolve cookie cache overwriting issues ...`).
