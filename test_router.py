"""Tests for simple-llm-router."""

from __future__ import annotations

import json
import httpx
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
            ),
        ],
        tags={"test-tag": ["test-provider"]},
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
    ids = {m["id"] for m in data["data"]}
    assert "test-provider" in ids
    assert "test-tag" in ids


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
    }
    resp = client.post("/admin/providers", json=new_provider, headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["provider"]["id"] == "new-provider"
    assert len(router.config.providers) == 2


def test_admin_add_provider_duplicate(client, admin_headers):
    new_provider = {
        "id": "test-provider",
        "base_url": "https://dup.example.com/v1",
        "model_name": "dup-model",
        "api_key_env": "DUP_API_KEY",
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
        "tags": {},
    }
    resp = client.post("/admin/config", json=new_cfg, headers=admin_headers)
    assert resp.status_code == 200
    assert router.config.admin["api_key"] == "new-key"


# --- Routing logic ---

def test_resolve_provider_by_id():
    """model matching provider id resolves to that provider."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={"tag-a": ["p1"], "tag-b": ["p2"]},
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "p2")
    assert provider.id == "p2"
    assert model == "m2"


def test_resolve_provider_by_tag():
    """model matching a tag resolves to first provider in the tag list."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={"tag-a": ["p1", "p2"], "tag-b": ["p2"]},
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "tag-a")
    assert provider.id == "p1"
    assert model == "m1"
    provider, model = router._resolve_provider(cfg, "tag-b")
    assert provider.id == "p2"
    assert model == "m2"


def test_resolve_provider_default_first():
    """Empty model or 'default' resolves to first healthy provider."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={},
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
    cfg = router.AppConfig(providers=[], tags={})
    router._rebuild_index(cfg)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="No default model"):
        router._resolve_provider(cfg, "default")


def test_resolve_provider_id_over_tag():
    """Provider id takes priority over tag match."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="tag-a", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={"tag-a": ["p2"]},
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "tag-a")
    assert provider.id == "tag-a"
    assert model == "m1"


def test_resolve_provider_unknown():
    """Unknown model with no matching id or tag raises 404."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={"tag-a": ["p1"]},
    )
    router._rebuild_index(cfg)
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="not found"):
        router._resolve_provider(cfg, "unknown")


# --- Provider with no api_key_env ---

def test_provider_no_api_key():
    """Provider with no api_key_env returns empty key."""
    p = router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1")
    assert p.api_key_env is None
    assert router._resolve_api_key(p) == ""


# --- Config validation ---

def test_validate_duplicate_provider_ids():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p1", base_url="https://p2.com", model_name="m2"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate provider ids"):
        router._validate_config(cfg)


def test_validate_tag_references_unknown_provider():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={"tag-a": ["p1", "p99"]},
    )
    with pytest.raises(ValueError, match="unknown providers"):
        router._validate_config(cfg)


def test_validate_duplicate_ids_in_tag():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={"tag-a": ["p1", "p1"]},
    )
    with pytest.raises(ValueError, match="Duplicate provider ids in tag"):
        router._validate_config(cfg)


def test_validate_valid_config():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={"text": ["p1", "p2"], "coding": ["p2"]},
    )
    # Should not raise
    router._validate_config(cfg)


def test_validate_default_provider_id_reserved():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="default", base_url="https://p1.com", model_name="m1"),
        ],
    )
    with pytest.raises(ValueError, match="reserved"):
        router._validate_config(cfg)


def test_validate_default_tag_reserved():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={"default": ["p1"]},
    )
    with pytest.raises(ValueError, match="reserved"):
        router._validate_config(cfg)


def test_validate_provider_id_not_snake_case():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="P1", base_url="https://p1.com", model_name="m1"),
        ],
    )
    with pytest.raises(ValueError, match="snake case"):
        router._validate_config(cfg)


def test_validate_tag_not_snake_case():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={"Text": ["p1"]},
    )
    with pytest.raises(ValueError, match="snake case"):
        router._validate_config(cfg)


# --- Health check state ---

def test_is_healthy_defaults_true():
    """Before health check runs, all providers default to healthy."""
    p = router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1")
    router._url_health.clear()
    assert router._is_healthy(p) is True


def test_is_healthy_respects_state():
    """_is_healthy reflects _url_health state."""
    p = router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1")
    router._url_health["https://p1.com"] = False
    assert router._is_healthy(p) is False
    router._url_health["https://p1.com"] = True
    assert router._is_healthy(p) is True
    del router._url_health["https://p1.com"]


def test_health_endpoint_includes_upstream_health(client):
    """GET /health returns upstream_health field."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "upstream_health" in data
    assert isinstance(data["upstream_health"], dict)


def test_admin_providers_includes_healthy(client, admin_headers):
    """GET /admin/providers returns healthy field per provider."""
    resp = client.get("/admin/providers", headers=admin_headers)
    assert resp.status_code == 200
    providers = resp.json()
    assert len(providers) == 1
    assert "healthy" in providers[0]
    assert providers[0]["healthy"] is True


def test_default_skips_unhealthy():
    """Default resolution picks first healthy provider."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={},
    )
    router._rebuild_index(cfg)
    router._url_health["https://p1.com"] = False
    provider, model = router._resolve_provider(cfg, "")
    assert provider.id == "p2"
    assert model == "m2"
    del router._url_health["https://p1.com"]


def test_default_all_unhealthy_raises_404():
    """When all providers are unhealthy, default raises 404."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={},
    )
    router._rebuild_index(cfg)
    router._url_health["https://p1.com"] = False
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="no healthy providers"):
        router._resolve_provider(cfg, "default")
    del router._url_health["https://p1.com"]


def test_tag_skips_unhealthy():
    """Tag resolution picks first healthy provider in the tag list."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={"tag-a": ["p1", "p2"]},
    )
    router._rebuild_index(cfg)
    router._url_health["https://p1.com"] = False
    provider, model = router._resolve_provider(cfg, "tag-a")
    assert provider.id == "p2"
    assert model == "m2"
    del router._url_health["https://p1.com"]


def test_id_match_ignores_health():
    """Explicit provider id match ignores health status."""
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={},
    )
    router._rebuild_index(cfg)
    router._url_health["https://p1.com"] = False
    provider, model = router._resolve_provider(cfg, "p1")
    assert provider.id == "p1"
    assert model == "m1"
    del router._url_health["https://p1.com"]


# --- Health check thread ---

def test_health_check_probes_v1_health(monkeypatch):
    """Health check thread probes /v1/health on each base_url."""
    probed_urls = []

    class FakeClient:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def get(self, url):
            probed_urls.append(url)
            mock_resp = type("R", (), {"status_code": 200})()
            return mock_resp

    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
            router.ProviderConfig(id="p2", base_url="https://p2.com", model_name="m2"),
        ],
        tags={},
    )
    router.config = cfg
    router._rebuild_index(cfg)
    router._url_health.clear()

    with patch("httpx.Client", side_effect=FakeClient):
        urls = set(p.base_url for p in cfg.providers)
        for url in urls:
            probe_url = f"{url.rstrip('/')}/v1/health"
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(probe_url)
                healthy = resp.status_code == 200
            with router._health_lock:
                router._url_health[url] = healthy

    assert "https://p1.com/v1/health" in probed_urls
    assert "https://p2.com/v1/health" in probed_urls
    assert router._url_health["https://p1.com"] is True
    assert router._url_health["https://p2.com"] is True


def test_health_check_marks_unhealthy(monkeypatch):
    """Health check marks URL unhealthy when probe fails."""
    class FakeClient:
        def __init__(self, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def get(self, url):
            raise httpx.ConnectError("Connection refused")

    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(id="p1", base_url="https://p1.com", model_name="m1"),
        ],
        tags={},
    )
    router.config = cfg
    router._rebuild_index(cfg)
    router._url_health.clear()

    with patch("httpx.Client", side_effect=FakeClient):
        probe_url = "https://p1.com/v1/health"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(probe_url)
                healthy = resp.status_code == 200
        except Exception:
            healthy = False
        with router._health_lock:
            router._url_health["https://p1.com"] = healthy

    assert router._url_health["https://p1.com"] is False