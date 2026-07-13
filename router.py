"""
simple-llm-router — single-file LLM provider router.

Exposes OpenAI-compatible API endpoints that proxy requests to
configurable upstream providers.  Admin endpoints allow runtime
configuration changes.

Usage:
    uvicorn router:app --host 127.0.0.1 --port 1135
    python -m router          # via pyproject.toml [project.scripts]
"""

from __future__ import annotations


import base64
import json
import logging
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("router")

# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    id: str
    base_url: str
    model_name: str
    api_key_env: Optional[str] = None


class AppConfig(BaseModel):
    # Defaults to empty api_key
    admin: dict[str, str] = Field(default_factory=lambda: {"api_key": ""})
    providers: list[ProviderConfig] = []
    tags: dict[str, list[str]] = Field(default_factory=dict)

# ---------------------------------------------------------------------------
# Config loader / store (in-memory, hot-reloadable via admin API)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(os.environ.get("ROUTER_CONFIG", str(Path(__file__).parent / "config.yaml")))


def _load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load config from YAML file with validation."""
    if not path.exists():
        log.warning("Config %s not found, using empty config", path)
        return AppConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = AppConfig(**data)
    _validate_config(cfg)
    _rebuild_index(cfg)
    return cfg


def _validate_config(cfg: AppConfig) -> None:
    """Validate config consistency. Raises ValueError on failure."""
    import re
    _snake_re = re.compile(r"^[a-z][a-z0-9]*([-_][a-z0-9]+)*$")
    provider_ids = {p.id for p in cfg.providers}

    # No duplicate provider ids
    if len(provider_ids) != len(cfg.providers):
        seen = set()
        dups = set()
        for p in cfg.providers:
            if p.id in seen:
                dups.add(p.id)
            seen.add(p.id)
        raise ValueError(f"Duplicate provider ids: {dups}")

    # Provider id: lowercase snake case, not "default"
    for p in cfg.providers:
        if p.id == "default":
            raise ValueError("Provider id 'default' is reserved")
        if not _snake_re.match(p.id):
            raise ValueError(f"Provider id '{p.id}' must be lowercase snake case")

    # No duplicate tag keys
    if len(cfg.tags) != len(set(cfg.tags.keys())):
        raise ValueError("Duplicate tag keys in tags section")

    for tag_name, tag_ids in cfg.tags.items():
        # Tag name: lowercase snake case, not "default"
        if tag_name == "default":
            raise ValueError("Tag name 'default' is reserved")
        if not _snake_re.match(tag_name):
            raise ValueError(f"Tag name '{tag_name}' must be lowercase snake case")

        # No duplicates within a tag's provider list
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError(f"Duplicate provider ids in tag '{tag_name}'")
        # All provider ids in tag must exist
        missing = set(tag_ids) - provider_ids
        if missing:
            raise ValueError(f"Tag '{tag_name}' references unknown providers: {missing}")


# Lookup tables — rebuilt on every config change
_id_to_provider: dict[str, ProviderConfig] = {}
_tag_to_providers: dict[str, list[str]] = {}

# Health check state — grouped by base_url
_url_health: dict[str, bool] = {}  # base_url → healthy?
_health_lock = threading.Lock()
_health_stop = threading.Event()
_health_thread: threading.Thread | None = None
HEALTH_CHECK_INTERVAL = 60  # seconds


def _is_healthy(provider: ProviderConfig) -> bool:
    """Check if a provider's base_url is currently healthy."""
    with _health_lock:
        return _url_health.get(provider.base_url, True)


def _run_health_checks() -> None:
    """Background thread: probe each unique base_url every HEALTH_CHECK_INTERVAL seconds."""
    while not _health_stop.is_set():
        # Collect unique base_urls from current config
        urls = set()
        for p in config.providers:
            urls.add(p.base_url)

        for url in urls:
            probe_url = f"{url.rstrip('/')}/health"
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(probe_url)
                    healthy = resp.status_code == 200
            except Exception as e:
                log.debug("Health check failed for %s: %s", probe_url, e)
                healthy = False

            with _health_lock:
                old = _url_health.get(url)
                _url_health[url] = healthy
                if old is not None and old != healthy:
                    log.info("Health changed for %s: %s → %s", url, "healthy" if old else "unhealthy", "healthy" if healthy else "unhealthy")
                elif old is None:
                    log.info("Health initial for %s: %s", url, "healthy" if healthy else "unhealthy")

        # Sleep in small increments so the stop event is responsive
        for _ in range(HEALTH_CHECK_INTERVAL):
            if _health_stop.is_set():
                break
            time.sleep(1)

    log.info("Health check thread stopped")


def _rebuild_index(cfg: AppConfig) -> None:
    """Build id->provider and tag->[provider_ids] lookups."""
    global _id_to_provider, _tag_to_providers
    _id_to_provider = {}
    _tag_to_providers = {}
    for p in cfg.providers:
        _id_to_provider[p.id] = p
    for tag_name, tag_ids in cfg.tags.items():
        _tag_to_providers[tag_name] = tag_ids


def _resolve_api_key(provider: ProviderConfig) -> str:
    """Resolve provider API key from environment. Empty string if no api_key_env."""
    if provider.api_key_env is None:
        return ""
    return os.environ.get(provider.api_key_env, "")


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _resolve_provider(cfg: AppConfig, model: str) -> tuple[ProviderConfig, str]:
    """
    Given a model name, find which provider to route to and what upstream
    model name to use.

    Resolution order:
    1. No model specified (empty string) → first healthy provider
    2. model matches a provider id → that provider (health ignored)
    3. model matches a tag → first healthy provider with that tag
    4. Otherwise → 404

    Returns (provider_config, upstream_model).
    Raises HTTPException(404) if no provider found.
    """
    # 0. No model specified → first healthy provider
    if not model or model == "default":
        for p in cfg.providers:
            if _is_healthy(p):
                return p, p.model_name
        raise HTTPException(
            status_code=404,
            detail="No default model: no healthy providers configured",
        )

    # 1. Check provider id (explicit selection — health ignored)
    provider = _id_to_provider.get(model)
    if provider:
        return provider, provider.model_name

    # 2. Check tags — first healthy provider with that tag
    provider_ids = _tag_to_providers.get(model)
    if provider_ids:
        for pid in provider_ids:
            provider = _id_to_provider.get(pid)
            if provider and _is_healthy(provider):
                return provider, provider.model_name

    # 3. Not found
    raise HTTPException(
        status_code=404,
        detail=f"Model '{model}' not found (no matching provider id or tag)",
    )


# ---------------------------------------------------------------------------
# HTTP proxy helpers (framework-agnostic)
# ---------------------------------------------------------------------------

def _prepare_upstream(
    provider: ProviderConfig,
    path: str,
    body: bytes,
    upstream_model: Optional[str] = None,
) -> tuple[str, dict[str, str], bytes]:
    """Build upstream URL, headers, and patched body. No FastAPI dependency."""
    api_key = _resolve_api_key(provider)
    upstream_url = f"{provider.base_url.rstrip('/')}/{path.lstrip('/')}"

    upstream_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        upstream_headers["Authorization"] = f"Bearer {api_key}"

    patched_body = body
    payload = json.loads(body)
    if upstream_model and upstream_model != payload.get("model"):
        payload["model"] = upstream_model
        patched_body = json.dumps(payload).encode()

    return upstream_url, upstream_headers, patched_body


async def _proxy_request(
    provider: ProviderConfig,
    path: str,
    method: str,
    body: bytes,
    upstream_model: Optional[str] = None,
) -> tuple[int, dict[str, str], bytes]:
    """
    Forward request to upstream. Returns (status_code, headers, body_bytes).
    No FastAPI dependency.
    """
    url, upstream_headers, patched_body = _prepare_upstream(provider, path, body, upstream_model)
    payload = json.loads(patched_body)
    is_stream = payload.get("stream", False)
    t0 = time.monotonic()

    if is_stream:
        full_body = b""
        status = 200
        resp_headers: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(method, url, headers=upstream_headers, content=patched_body) as resp:
                status = resp.status_code
                resp_headers = dict(resp.headers)
                latency_ms = (time.monotonic() - t0) * 1000
                log.info(
                    "proxied %s %s -> provider=%s model=%s status=%d latency=%.0fms",
                    method, path, provider.id,
                    upstream_model or payload.get("model", "?"),
                    status, latency_ms,
                )
                async for chunk in resp.aiter_bytes():
                    full_body += chunk
        return status, resp_headers, full_body

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            method=method, url=url, headers=upstream_headers, content=patched_body,
        )
    latency_ms = (time.monotonic() - t0) * 1000
    log.info(
        "proxied %s %s -> provider=%s model=%s status=%d latency=%.0fms",
        method, path, provider.id,
        upstream_model or payload.get("model", "?"),
        resp.status_code, latency_ms,
    )
    exclude_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in exclude_headers}
    return resp.status_code, resp_headers, resp.content


async def _proxy_request_stream(
    provider: ProviderConfig,
    path: str,
    method: str,
    body: bytes,
    upstream_model: Optional[str] = None,
):
    """Async generator yielding raw SSE bytes. No FastAPI dependency."""
    url, upstream_headers, patched_body = _prepare_upstream(provider, path, body, upstream_model)
    payload = json.loads(patched_body)
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(method, url, headers=upstream_headers, content=patched_body) as resp:
            latency_ms = (time.monotonic() - t0) * 1000
            log.info(
                "proxied %s %s -> provider=%s model=%s status=%d latency=%.0fms",
                method, path, provider.id,
                upstream_model or payload.get("model", "?"),
                resp.status_code, latency_ms,
            )
            async for chunk in resp.aiter_bytes():
                yield chunk


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# Global config — reloaded via admin API
config: AppConfig = _load_config()
_config_path: Path = CONFIG_PATH


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _health_thread
    log.info("Config: %s", _config_path.resolve())
    log.info("Loaded %d provider(s): %s", len(config.providers), [p.id for p in config.providers])
    for p in config.providers:
        log.info("  provider %s: %s (model=%s)", p.id, p.base_url, p.model_name)
    if config.tags:
        log.info("Tags: %s", list(config.tags.keys()))

    # Start health check background thread
    _health_stop.clear()
    _health_thread = threading.Thread(target=_run_health_checks, daemon=True, name="health-check")
    _health_thread.start()
    log.info("Health check thread started (interval=%ds)", HEALTH_CHECK_INTERVAL)

    yield

    # Stop health check thread
    _health_stop.set()
    if _health_thread is not None:
        _health_thread.join(timeout=HEALTH_CHECK_INTERVAL + 2)
        _health_thread = None


app = FastAPI(
    title="simple-llm-router",
    version="0.1.0",
    description="Lightweight LLM provider router with OpenAI-compatible API",
    lifespan=lifespan,
)


# --- Health ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": len(config.providers),
        "models": len(config.providers),
        "upstream_health": dict(_url_health),
    }


# --- Admin endpoints ---

def _check_admin_key(authorization: str | None) -> None:
    """Validate admin API key. Raises 401 on failure."""
    expected = config.admin.get("api_key", "change-me")
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Admin API key required")


def _check_api_key(authorization: str | None) -> None:
    """Validate API key for OpenAI-compatible endpoints.

    If config admin api_key is empty (or default 'change-me'), skip verification.
    Otherwise, require Authorization: Bearer ***
    """
    expected = config.admin.get("api_key", "change-me")
    if not expected:
        return
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="API key required")


@app.get("/admin/config")
async def get_config(authorization: str | None = Header(None)):
    """Get current runtime configuration."""
    _check_admin_key(authorization)
    return config.model_dump()


@app.post("/admin/config")
async def update_config(
    new_config: dict[str, Any],
    authorization: str | None = Header(None),
):
    """Update runtime configuration. Changes take effect immediately."""
    _check_admin_key(authorization)
    global config, _config_path
    config = AppConfig(**new_config)
    _validate_config(config)
    _config_path = Path("<admin API>")
    _rebuild_index(config)
    log.info("Config updated via admin API")
    return {"status": "ok", "message": "configuration updated"}


@app.get("/admin/providers")
async def list_providers(authorization: str | None = Header(None)):
    """List all configured providers with health status."""
    _check_admin_key(authorization)
    result = []
    for p in config.providers:
        info = p.model_dump()
        info["healthy"] = _is_healthy(p)
        result.append(info)
    return result


@app.post("/admin/providers")
async def add_provider(
    provider: ProviderConfig,
    authorization: Optional[str] = Header(None),
):
    """Add a new provider at runtime."""
    _check_admin_key(authorization)
    global config
    if provider.id in _id_to_provider:
        raise HTTPException(status_code=409, detail=f"Provider '{provider.id}' already exists")
    config.providers.append(provider)
    _rebuild_index(config)
    log.info("Provider '%s' added via admin API", provider.id)
    return {"status": "ok", "provider": provider.model_dump()}


@app.delete("/admin/providers/{provider_id}")
async def remove_provider(
    provider_id: str,
    authorization: Optional[str] = Header(None),
):
    """Remove a provider by id."""
    _check_admin_key(authorization)
    global config
    if provider_id not in _id_to_provider:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")
    config.providers = [p for p in config.providers if p.id != provider_id]
    _rebuild_index(config)
    log.info("Provider '%s' removed via admin API", provider_id)
    return {"status": "ok", "removed": provider_id}


# --- OpenAI-compatible endpoints ---

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """List all available models across configured providers."""
    models = []
    seen = set()
    for p in config.providers:
        if p.id not in seen:
            models.append({
                "id": p.id,
                "object": "model",
                "created": 0,
                "owned_by": p.id,
            })
            seen.add(p.id)
    for tag_name in config.tags:
        if tag_name not in seen:
            models.append({
                "id": tag_name,
                "object": "model",
                "created": 0,
                "owned_by": tag_name,
            })
            seen.add(tag_name)
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy chat completion request to upstream provider."""
    _check_api_key(authorization)
    body = await request.body()
    payload = json.loads(body)
    model = payload.get("model", "default")

    provider, upstream_model = _resolve_provider(config, model)

    if payload.get("stream", False):
        return StreamingResponse(
            _proxy_request_stream(provider, "chat/completions", "POST", body, upstream_model),
            media_type="text/event-stream",
        )
    status, resp_headers, resp_body = await _proxy_request(provider, "chat/completions", "POST", body, upstream_model)
    return Response(content=resp_body, status_code=status, headers=resp_headers)


@app.post("/v1/completions")
@app.post("/completions")
async def completions(
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy legacy completion request to upstream provider."""
    _check_api_key(authorization)
    body = await request.body()
    payload = json.loads(body)
    model = payload.get("model", "default")

    provider, upstream_model = _resolve_provider(config, model)

    if payload.get("stream", False):
        return StreamingResponse(
            _proxy_request_stream(provider, "completions", "POST", body, upstream_model),
            media_type="text/event-stream",
        )
    status, resp_headers, resp_body = await _proxy_request(provider, "completions", "POST", body, upstream_model)
    return Response(content=resp_body, status_code=status, headers=resp_headers)


# Catch-all: forward any /v1/... or /... path
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all_proxy(
    request: Request,
    path: str,
    authorization: str | None = Header(None),
):
    """
    Catch-all proxy: forward unrecognized paths to the first configured provider.
    Useful for endpoints we haven't explicitly defined.
    """
    _check_api_key(authorization)
    if path.startswith("admin"):
        raise HTTPException(status_code=404, detail=f"Admin endpoint '/{path}' not found")

    body = await request.body()
    payload = {}
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        pass

    model = payload.get("model", "")
    if model:
        provider, upstream_model = _resolve_provider(config, model)
    elif config.providers:
        provider = config.providers[0]
        upstream_model = provider.model_name
    else:
        raise HTTPException(status_code=503, detail="No providers configured")

    if payload.get("stream", False):
        return StreamingResponse(
            _proxy_request_stream(provider, path, request.method, body, upstream_model or None),
            media_type="text/event-stream",
        )
    status, resp_headers, resp_body = await _proxy_request(provider, path, request.method, body, upstream_model or None)
    return Response(content=resp_body, status_code=status, headers=resp_headers)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _cli_chat(config_path: Path, model: str | None, message: str, image_path: str | None = None) -> None:
    """Start FastAPI app, call its own API, print response, exit."""
    import socket
    import sys
    import threading
    import uvicorn

    # Export config path so the app picks it up
    os.environ["ROUTER_CONFIG"] = str(config_path)

    # Reload global config from the specified path
    global config
    config = _load_config(config_path)
    _config_path = config_path

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    base_url = f"http://127.0.0.1:{port}"

    # Start uvicorn in background thread
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    with httpx.Client(timeout=5.0) as client:
        for _ in range(60):
            try:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(0.1)
        else:
            print("Server startup timeout", file=sys.stderr)
            sys.exit(1)

        # Build request content
        if image_path:
            img_file = Path(image_path)
            if not img_file.exists():
                print(f"Image not found: {image_path}", file=sys.stderr)
                sys.exit(1)
            mime, _ = mimetypes.guess_type(image_path)
            if not mime:
                mime = "image/png"
            raw = img_file.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"
            content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            content = message

        # Build request — let the API resolve model (default → fallback_order)
        payload = {
            "model": model or "default",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }

        resp = client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0,
        )

        with resp as r:
            if r.status_code != 200:
                print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
                sys.exit(1)

            # Parse SSE stream
            reasoning_started = False
            reasoning_ended = False

            def _flush_reasoning():
                nonlocal reasoning_started, reasoning_ended
                if reasoning_started and not reasoning_ended:
                    sys.stdout.write("[/thinking]\n")
                    sys.stdout.flush()
                    reasoning_ended = True

            def _emit(delta: dict):
                nonlocal reasoning_started, reasoning_ended
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    if not reasoning_started:
                        sys.stdout.write("\n[thinking] ")
                        sys.stdout.flush()
                        reasoning_started = True
                    sys.stdout.write(reasoning)
                    sys.stdout.flush()
                content = delta.get("content", "")
                if content:
                    _flush_reasoning()
                    sys.stdout.write(content)
                    sys.stdout.flush()

            buffer = b""
            for chunk in r.iter_bytes(chunk_size=1024):
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        _flush_reasoning()
                        print()
                        break
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            print(f"Error: {data['error']}", file=sys.stderr)
                            sys.exit(1)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        _emit(delta)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

            _flush_reasoning()
            print()

    # Shutdown
    server.should_exit = True
    thread.join(timeout=3)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="router",
        description="simple-llm-router: proxy server or CLI chat client",
    )
    parser.add_argument(
        "-c", "--config",
        metavar="CONFIG",
        default=str(CONFIG_PATH),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "-p", "--provider",
        metavar="MODEL",
        default=None,
        help="Model alias to use (default: first model in config)",
    )
    parser.add_argument(
        "-m", "--message",
        metavar="MESSAGE",
        default=None,
        help="Message to send to the selected model (CLI mode; omit for server mode)",
    )
    parser.add_argument(
        "-f", "--file",
        metavar="IMAGE",
        default=None,
        help="Path to an image file to include with the message (CLI mode)",
    )
    args = parser.parse_args()

    if args.message:
        _cli_chat(Path(args.config), args.provider, args.message, args.file)
    else:
        # Export config path via env var so uvicorn subprocess picks it up
        custom_config = Path(args.config)
        os.environ["ROUTER_CONFIG"] = str(custom_config)
        import uvicorn
        uvicorn.run(
            "router:app",
            host="127.0.0.1",
            port=1135,
            log_level="info",
        )


if __name__ == "__main__":
    main()
