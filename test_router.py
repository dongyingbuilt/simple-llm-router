"""Tests for simple-llm-router."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import router


@pytest.fixture(autouse=True)
def _fresh_config(tmp_path):
    """Load a minimal config for each test."""
    cfg = router.AppConfig(
        admin={"api_key": "test-key"},
        providers=[
            router.ProviderConfig(
                id="test-provider",
                base_url="https://test.example.com/v1",
                model_name="test-model",
                api_key_env="TEST_API_KEY",
                tags=["test-tag"],
            ),
        ],
    )
    router.config = cfg
    router._rebuild_index(cfg)
    return cfg


@pytest.fixture
def client():
    return TestClient(router.app)


@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test-key"}


@pytest.fixture
def api_headers():
    return {"Authorization": "Bearer test-key"}


# --- Health ---

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["providers"] == 1
    assert data["models"] == 1


# --- Models listing ---

def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    # Should have provider id + tags
    ids = {m["id"] for m in data["data"]}
    assert "test-provider" in ids
    assert "test-tag" in ids
    assert data["data"][0]["owned_by"] == "test-provider"


def test_list_models_short_path(client):
    resp = client.get("/models")
    assert resp.status_code == 200


# --- Chat completions proxy ---

def test_chat_completions_proxy(client, monkeypatch, api_headers):
    """POST /v1/chat/completions should proxy to upstream."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "chat-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([mock_resp.content])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"model": "test-provider", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body, headers=api_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello"


def test_chat_completions_streaming(client, monkeypatch, api_headers):
    """POST /v1/chat/completions with stream=true should return SSE."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    sse_chunks = [
        b'data: {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"content": " World"}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    class FakeStream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            for c in sse_chunks:
                yield c

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def stream(self, *a, **kw):
            return FakeStream()

    with patch("httpx.AsyncClient", side_effect=FakeClient):
        body = {
            "model": "test-provider",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=body, headers=api_headers)

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    # All chunks should be present in the streamed response
    assert '"content": "Hello"' in text
    assert '"content": " World"' in text
    assert '"reasoning_content": "thinking"' in text
    assert "[DONE]" in text


def test_chat_completions_nonstream_reasoning(client, monkeypatch, api_headers):
    """Non-streaming response with reasoning_content should pass through."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "chat-1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "Let me think",
                },
            }],
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([mock_resp.content])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"model": "test-provider", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body, headers=api_headers)

    assert resp.status_code == 200
    data = resp.json()
    msg = data["choices"][0]["message"]
    assert msg["content"] == "Answer"
    assert msg["reasoning_content"] == "Let me think"


def test_chat_completions_missing_model_uses_default(client, monkeypatch, api_headers):
    """When model is missing, picks first provider."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "chat-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([mock_resp.content])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body, headers=api_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello"


def test_chat_completions_explicit_default_model(client, monkeypatch, api_headers):
    """Explicit model='default' resolves to first provider."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "chat-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([mock_resp.content])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body, headers=api_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello"


def test_chat_completions_unknown_model(client, api_headers):
    """unknown model with no matching id or tag should return 404."""
    body = {"model": "unknown-model", "messages": []}
    resp = client.post("/v1/chat/completions", json=body, headers=api_headers)
    assert resp.status_code == 404


def test_chat_completions_no_auth_when_api_key_empty(client, monkeypatch):
    """When admin api_key is empty, chat completions should not require auth."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")
    router.config.admin["api_key"] = ""

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({
            "id": "chat-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
        }).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([mock_resp.content])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"model": "test-provider", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello"


# --- Admin endpoints ---

def test_admin_config_get(client, admin_headers):
    resp = client.get("/admin/config", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["providers"]) == 1


def test_admin_config_no_auth(client):
    resp = client.get("/admin/config")
    assert resp.status_code == 401


def test_admin_providers_list(client, admin_headers):
    resp = client.get("/admin/providers", headers=admin_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_admin_add_provider(client, admin_headers):
    new_provider = {
        "id": "new-provider",
        "base_url": "https://new.example.com/v1",
        "model_name": "new-model",
        "api_key_env": "NEW_API_KEY",
        "tags": ["new-tag"],
    }
    resp = client.post("/admin/providers", json=new_provider, headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["provider"]["id"] == "new-provider"
    # Verify it's in the live config
    assert len(router.config.providers) == 2


def test_admin_add_provider_duplicate(client, admin_headers):
    new_provider = {
        "id": "test-provider",
        "base_url": "https://dup.example.com/v1",
        "model_name": "dup-model",
        "api_key_env": "DUP_API_KEY",
        "tags": [],
    }
    resp = client.post("/admin/providers", json=new_provider, headers=admin_headers)
    assert resp.status_code == 409


def test_admin_remove_provider(client, admin_headers):
    resp = client.delete("/admin/providers/test-provider", headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["removed"] == "test-provider"
    assert len(router.config.providers) == 0


def test_admin_remove_provider_not_found(client, admin_headers):
    resp = client.delete("/admin/providers/nonexistent", headers=admin_headers)
    assert resp.status_code == 404


def test_admin_update_config(client, admin_headers):
    new_cfg = {
        "admin": {"api_key": "new-key"},
        "providers": [],
    }
    resp = client.post("/admin/config", json=new_cfg, headers=admin_headers)
    assert resp.status_code == 200
    assert router.config.admin["api_key"] == "new-key"


# --- Routing logic ---

def test_resolve_provider_by_id():
    """model matching provider id resolves to that provider."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", base_url="https://p1.com",
                model_name="m1", tags=["tag-a"],
            ),
            router.ProviderConfig(
                id="p2", base_url="https://p2.com",
                model_name="m2", tags=["tag-b"],
            ),
        ],
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "p2")
    assert provider.id == "p2"
    assert model == "m2"


def test_resolve_provider_by_tag():
    """model matching a tag resolves to first provider with that tag."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", base_url="https://p1.com",
                model_name="m1", tags=["tag-a"],
            ),
            router.ProviderConfig(
                id="p2", base_url="https://p2.com",
                model_name="m2", tags=["tag-a", "tag-b"],
            ),
        ],
    )
    router._rebuild_index(cfg)
    # tag-a should resolve to p1 (first in list)
    provider, model = router._resolve_provider(cfg, "tag-a")
    assert provider.id == "p1"
    assert model == "m1"
    # tag-b should resolve to p2
    provider, model = router._resolve_provider(cfg, "tag-b")
    assert provider.id == "p2"
    assert model == "m2"


def test_resolve_provider_default_first():
    """Empty model or 'default' resolves to first provider."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", base_url="https://p1.com",
                model_name="m1", tags=[],
            ),
            router.ProviderConfig(
                id="p2", base_url="https://p2.com",
                model_name="m2", tags=[],
            ),
        ],
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "")
    assert provider.id == "p1"
    assert model == "m1"
    provider, model = router._resolve_provider(cfg, "default")
    assert provider.id == "p1"
    assert model == "m1"


def test_resolve_provider_default_empty_providers():
    """Empty model with no providers raises 404."""
    cfg = router.AppConfig(providers=[])
    router._rebuild_index(cfg)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="No default model"):
        router._resolve_provider(cfg, "default")


def test_resolve_provider_id_over_tag():
    """Provider id takes priority over tag match."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="tag-a", base_url="https://p1.com",
                model_name="m1", tags=["tag-b"],
            ),
            router.ProviderConfig(
                id="p2", base_url="https://p2.com",
                model_name="m2", tags=["tag-a"],
            ),
        ],
    )
    router._rebuild_index(cfg)
    # "tag-a" matches provider id first, not tag
    provider, model = router._resolve_provider(cfg, "tag-a")
    assert provider.id == "tag-a"
    assert model == "m1"


def test_resolve_provider_unknown():
    """Unknown model with no matching id or tag raises 404."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", base_url="https://p1.com",
                model_name="m1", tags=["tag-a"],
            ),
        ],
    )
    router._rebuild_index(cfg)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="not found"):
        router._resolve_provider(cfg, "unknown")


# --- Provider with no api_key_env ---

def test_provider_no_api_key():
    """Provider with no api_key_env returns empty key."""
    p = router.ProviderConfig(
        id="p1", base_url="https://p1.com",
        model_name="m1", tags=[],
    )
    assert p.api_key_env is None
    assert router._resolve_api_key(p) == ""
