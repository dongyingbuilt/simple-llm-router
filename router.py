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


import json
import logging
import os
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
    name: str
    base_url: str
    api_key_env: str
    models: list[str] = []


class RoutingConfig(BaseModel):
    rules: dict[str, dict[str, str]] = {}
    fallback_order: list[str] = []


class AppConfig(BaseModel):
    admin: dict[str, str] = Field(default_factory=lambda: {"api_key": "change-me"})
    providers: list[ProviderConfig] = []
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

# ---------------------------------------------------------------------------
# Config loader / store (in-memory, hot-reloadable via admin API)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(os.environ.get("ROUTER_CONFIG", str(Path(__file__).parent / "config.yaml")))


def _load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load config from YAML file."""
    if not path.exists():
        log.warning("Config %s not found, using empty config", path)
        return AppConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    cfg = AppConfig(**data)
    # Build lookup tables
    _rebuild_index(cfg)
    return cfg


# Lookup tables — rebuilt on every config change
_model_to_providers: dict[str, list[str]] = {}
_id_to_provider: dict[str, ProviderConfig] = {}


def _rebuild_index(cfg: AppConfig) -> None:
    """Build model->provider and id->provider lookups."""
    global _model_to_providers, _id_to_provider
    _model_to_providers = {}
    _id_to_provider = {}
    for p in cfg.providers:
        _id_to_provider[p.id] = p
        for m in p.models:
            _model_to_providers.setdefault(m, []).append(p.id)


def _resolve_api_key(provider: ProviderConfig) -> str:
    """Resolve provider API key from environment."""
    key = os.environ.get(provider.api_key_env, "")
    return key


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _resolve_default_model(cfg: AppConfig) -> tuple[ProviderConfig, str]:
    """Resolve 'default' to the first provider in fallback_order and its first model."""
    for pid in cfg.routing.fallback_order:
        provider = _id_to_provider.get(pid)
        if provider and provider.models:
            return provider, provider.models[0]
    raise HTTPException(
        status_code=404,
        detail="No default model: fallback_order is empty or has no models",
    )


def _resolve_provider(cfg: AppConfig, model: str) -> tuple[ProviderConfig, str]:
    """
    Given a model name, find which provider to route to and what upstream
    model name to use.

    Returns (provider_config, upstream_model).
    Raises HTTPException(404) if no provider found.
    """
    # 0. 'default' resolves to first provider in fallback_order
    if model == "default":
        return _resolve_default_model(cfg)

    # 1. Check explicit routing rules
    if model in cfg.routing.rules:
        rule = cfg.routing.rules[model]
        pid = rule["provider"]
        upstream = rule.get("upstream_model", model)
        provider = _id_to_provider.get(pid)
        if provider:
            return provider, upstream

    # 2. Check model registry
    provider_ids = _model_to_providers.get(model, [])
    if provider_ids:
        return _id_to_provider[provider_ids[0]], model

    # 3. Fallback order
    for pid in cfg.routing.fallback_order:
        provider = _id_to_provider.get(pid)
        if provider:
            return provider, model

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model}' not found in any configured provider",
    )


# ---------------------------------------------------------------------------
# HTTP proxy helpers
# ---------------------------------------------------------------------------

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
    log.info("Config: %s", _config_path.resolve())
    log.info("Loaded %d provider(s): %s", len(config.providers), [p.id for p in config.providers])
    for p in config.providers:
        log.info("  provider %s (%s): %s — models: %s", p.id, p.name, p.base_url, p.models)
    if config.routing.rules:
        log.info("Routing rules: %s", config.routing.rules)
    if config.routing.fallback_order:
        log.info("Fallback order: %s", config.routing.fallback_order)
    yield


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
        "models": sum(len(p.models) for p in config.providers),
    }


# --- Admin endpoints ---

def _check_admin_key(authorization: str | None) -> None:
    """Validate admin API key. Raises 401 on failure."""
    expected = config.admin.get("api_key", "change-me")
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Admin API key required")


@app.get("/admin/config")
async def get_config(authorization: Optional[str] = Header(None)):
    """Get current runtime configuration."""
    _check_admin_key(authorization)
    return config.model_dump()


@app.post("/admin/config")
async def update_config(
    new_config: dict[str, Any],
    authorization: Optional[str] = Header(None),
):
    """Update runtime configuration. Changes take effect immediately."""
    _check_admin_key(authorization)
    global config, _config_path
    config = AppConfig(**new_config)
    _config_path = Path("<admin API>")
    _rebuild_index(config)
    log.info("Config updated via admin API")
    return {"status": "ok", "message": "configuration updated"}


@app.get("/admin/providers")
async def list_providers(authorization: Optional[str] = Header(None)):
    """List all configured providers."""
    _check_admin_key(authorization)
    return [p.model_dump() for p in config.providers]


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
        for m in p.models:
            if m not in seen:
                models.append({
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": p.id,
                })
                seen.add(m)
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy chat completion request to upstream provider."""
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
        upstream_model = model
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

def _cli_chat(config_path: Path, model: str | None, message: str) -> None:
    """Load config, resolve model, send message, print response, exit."""
    import asyncio
    import sys

    # Load the specified config
    cfg = _load_config(config_path)
    # Rebuild global index so _resolve_provider works
    _rebuild_index(cfg)

    # Resolve model: explicit > first provider in fallback_order > first provider
    if not model:
        if cfg.routing.fallback_order:
            fallback_id = cfg.routing.fallback_order[0]
            p = _id_to_provider.get(fallback_id)
            if p and p.models:
                model = p.models[0]
        if not model and cfg.providers:
            model = cfg.providers[0].models[0] if cfg.providers[0].models else None
        if not model:
            print("No default model available. Use -p to specify one.", file=sys.stderr)
            sys.exit(1)

    provider, upstream_model = _resolve_provider(cfg, model)

    payload = {
        "model": upstream_model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }
    body = json.dumps(payload).encode()

    print(f"model={model} provider={provider.id}", file=sys.stderr)

    async def _run() -> None:
        reasoning_started = False
        reasoning_ended = False

        def _flush_reasoning() -> None:
            nonlocal reasoning_started, reasoning_ended
            if reasoning_started and not reasoning_ended:
                sys.stdout.write("[/thinking]\n")
                sys.stdout.flush()
                reasoning_ended = True

        def _emit(delta: dict) -> None:
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
        error_status = None
        try:
            async for chunk in _proxy_request_stream(
                provider, "chat/completions", "POST", body, upstream_model
            ):
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
                        return
                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            error_status = 400
                            print(f"Error: {data['error']}", file=sys.stderr)
                            return
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        _emit(delta)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
            # Draining leftover buffer
            if buffer.strip():
                line_str = buffer.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            _emit(delta)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
            _flush_reasoning()
            print()
        except httpx.HTTPStatusError as e:
            print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_run())


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
    args = parser.parse_args()

    if args.message:
        _cli_chat(Path(args.config), args.provider, args.message)
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
