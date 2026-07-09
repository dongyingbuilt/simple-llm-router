# simple-llm-router

A lightweight LLM proxy that routes requests to configurable providers via a standard OpenAI-compatible API.

## Structure

Single-file Python project. Everything lives in `router.py`.

## Tech

- **Runtime**: Python 3.10+
- **Framework**: FastAPI + uvicorn (async)
- **HTTP client**: `httpx` for upstream provider calls
- **Config**: YAML or JSON config file for provider list and routing rules
- **Package manager**: `uv` or `pip`

## API

### OpenAI-compatible endpoints (request ingress)

- `POST /v1/chat/completions` — proxy to upstream provider
- `GET /v1/models` — list configured models across all providers

Requests are routed based on config: model name mapping, provider selection, fallback order.

### Admin endpoints (settings management)

- `GET /admin/config` — current configuration
- `POST /admin/config` — update configuration at runtime
- `GET /admin/providers` — list all configured providers
- `POST /admin/providers` — add a new provider
- `DELETE /admin/providers/{id}` — remove a provider

## Conventions

- Match OpenAI v1 API request/response schema exactly for the compatible endpoints
- Config changes take effect immediately without restart
- Log all routed requests with provider, model, and latency
- Keep dependencies minimal — one router.py file, one config.yaml file, one test_router.py file, no ORM, no database

## Dependencies

```
fastapi
uvicorn
httpx
pydantic
pyyaml
```

## Run

```bash
uvicorn router:app --host 127.0.0.0 --port 1135
```

## Test

```bash
pytest test_router.py -v
```
