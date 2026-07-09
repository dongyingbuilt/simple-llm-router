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
                name="Test",
                base_url="https://test.example.com/v1",
                api_key_env="TEST_API_KEY",
                models=["test-model"],
            ),
        ],
        routing=router.RoutingConfig(fallback_order=["test-provider"]),
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
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-model"
    assert data["data"][0]["owned_by"] == "test-provider"


def test_list_models_short_path(client):
    resp = client.get("/models")
    assert resp.status_code == 200


# --- Chat completions proxy ---

def test_chat_completions_proxy(client, monkeypatch):
    """POST /v1/chat/completions should proxy to upstream."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    async def fake_request(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"id": "chat-1", "choices": []}).encode()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.aiter_bytes = lambda: iter([json.dumps({"id": "chat-1", "choices": []}).encode()])
        return mock_resp

    with patch("httpx.AsyncClient.request", side_effect=fake_request):
        body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200


def test_chat_completions_missing_model(client):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert "model" in resp.json()["detail"].lower()


def test_chat_completions_unknown_model(client):
    """unknown model with no fallback should return 404."""
    # Temporarily clear fallback so unknown model truly 404s
    router.config.routing.fallback_order = []
    body = {"model": "unknown-model", "messages": []}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 404


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
        "name": "New",
        "base_url": "https://new.example.com/v1",
        "api_key_env": "NEW_API_KEY",
        "models": ["new-model"],
    }
    resp = client.post("/admin/providers", json=new_provider, headers=admin_headers)
    assert resp.status_code == 200
    assert resp.json()["provider"]["id"] == "new-provider"
    # Verify it's in the live config
    assert len(router.config.providers) == 2


def test_admin_add_provider_duplicate(client, admin_headers):
    new_provider = {
        "id": "test-provider",
        "name": "Dup",
        "base_url": "https://dup.example.com/v1",
        "api_key_env": "DUP_API_KEY",
        "models": [],
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
        "routing": {"rules": {}, "fallback_order": []},
    }
    resp = client.post("/admin/config", json=new_cfg, headers=admin_headers)
    assert resp.status_code == 200
    assert router.config.admin["api_key"] == "new-key"


# --- Routing logic ---

def test_resolve_provider_by_model():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", name="P1", base_url="https://p1.com",
                api_key_env="K1", models=["m1", "m2"],
            ),
            router.ProviderConfig(
                id="p2", name="P2", base_url="https://p2.com",
                api_key_env="K2", models=["m2", "m3"],
            ),
        ],
        routing=router.RoutingConfig(),
    )
    router._rebuild_index(cfg)
    # m1 only in p1
    provider, model = router._resolve_provider(cfg, "m1")
    assert provider.id == "p1"
    assert model == "m1"


def test_resolve_provider_with_rule():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", name="P1", base_url="https://p1.com",
                api_key_env="K1", models=["real-model"],
            ),
        ],
        routing=router.RoutingConfig(
            rules={"alias-model": {"provider": "p1", "upstream_model": "real-model"}},
        ),
    )
    router._rebuild_index(cfg)
    provider, model = router._resolve_provider(cfg, "alias-model")
    assert provider.id == "p1"
    assert model == "real-model"


def test_resolve_provider_fallback():
    cfg = router.AppConfig(
        providers=[
            router.ProviderConfig(
                id="p1", name="P1", base_url="https://p1.com",
                api_key_env="K1", models=["m1"],
            ),
        ],
        routing=router.RoutingConfig(fallback_order=["p1"]),
    )
    router._rebuild_index(cfg)
    # unknown model falls back to first provider in fallback_order
    provider, model = router._resolve_provider(cfg, "unknown")
    assert provider.id == "p1"
    assert model == "unknown"
