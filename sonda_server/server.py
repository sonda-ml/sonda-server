# SPDX-License-Identifier: Apache-2.0
"""The HTTP API (FastAPI on uvicorn): /v1/systemone, /health, /v1/model, /metrics, /admin/reload, and the page
with API examples and a game at / (with /demo/examples.json; off with --no-demo).

POST /v1/systemone — the TypeSafe-style decision API:

    request   {"state": ..., "questions": {"q1": {...}, ...}, "model": "..."?}
    response  {"model": "...", "answers": {"q1": {...}}, "usage": {"input_tokens": n, "output_tokens": 0},
               "latency_ms": t, "omitted": ["q2"]?}

Access: with an API key set (SONDA_API_KEY), /v1/systemone, /v1/model and /metrics (unless --public-metrics) need
`Authorization: Bearer <key>`; /health, the page and its examples stay open (they hold nothing secret). Every
request must name an allowed Host (DNS rebinding), and a POST sent by a web page of another origin is refused
(a page elsewhere cannot use a local server through the visitor's browser).

Status codes: 200 answered · 400 not JSON or a Host not served · 401 missing/wrong API key (only when SONDA_API_KEY is set) ·
403 a POST from a web page of another origin ·
404 unknown path · 413 body too large · 422 invalid request or a prompt over max_input_tokens (with
{"error": ..., "errors": [{"loc", "msg"}]}) · 503 queue full (Retry-After) · 504 over request_timeout_s ·
500 unexpected (no traceback is returned; it is logged).

Concurrency: the event loop only parses, validates and answers; each question runs in a request thread
(tokenizing and, for more than 16 options, the spread logic) and its passes go to the shared batcher, so
questions of the same request and of other requests are computed in the same forwards.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from urllib.parse import urlsplit

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import __version__
from .batcher import Overloaded
from .config import Settings
from .engine import InputTooLong
from .metrics import Metrics
from .schemas import filter_answer, validate_request

log = logging.getLogger("sonda_server")

WEB = files("sonda_server") / "web"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# The page is self-contained: inline script and style, requests only to this server.
PAGE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
                                "frame-ancestors 'none'"),
}


def _json(status: int, payload: dict, headers: dict | None = None) -> Response:
    # orjson writes NaN and infinity as null, so even an unfiltered answer is valid JSON.
    return Response(orjson.dumps(payload), status_code=status, media_type="application/json",
                    headers={"X-Content-Type-Options": "nosniff", **(headers or {})})


def host_name(netloc: str) -> str:
    """"Example.com:8090" -> "example.com", "[::1]:8090" -> "::1"."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1:netloc.find("]")] if "]" in netloc else netloc
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def allowed_host_names(settings: Settings) -> set[str] | None:
    """The Host names served; None means any."""
    listed = settings.allowed_hosts.strip()
    if listed == "*":
        return None
    if listed:
        return {host_name(h) for h in listed.split(",") if h.strip()}
    return set(LOOPBACK) if settings.host in LOOPBACK else None


def create_app(settings: Settings, engine, metrics: Metrics | None = None) -> FastAPI:
    """The application around an engine (engine.Engine, or any object with decide / check_lengths / reload /
    config, which the tests use)."""
    metrics = metrics or Metrics()
    pool = ThreadPoolExecutor(max_workers=settings.threads, thread_name_prefix="sonda-request")
    app = FastAPI(title="sonda-server", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.engine, app.state.metrics = settings, engine, metrics

    def validation_state() -> dict:
        return {"input": settings.input_validation, "output": settings.output_filter,
                "typesafe_compat": settings.typesafe_compat}

    def key_from_env(name: str) -> str:
        return os.environ.get(name, "") if name else ""

    def authorized(request: Request, env_name: str) -> bool:
        expected = key_from_env(env_name)
        if not expected:
            return True
        given = request.headers.get("authorization", "")
        return hmac.compare_digest(given.encode(), f"Bearer {expected}".encode())

    def reject(status: int, reason: str, message: str, errors: list | None = None, headers=None) -> Response:
        metrics.inc("rejected_total", reason=reason)
        metrics.inc("requests_total", status=str(status))
        log.info(json.dumps({"event": "rejected", "status": status, "reason": reason}))
        body = {"error": message}
        if errors:
            body["errors"] = errors
        return _json(status, body, headers)

    hosts = allowed_host_names(settings)
    listed = {host_name(h) for h in settings.allowed_hosts.split(",") if h.strip() and h.strip() != "*"}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if hosts is not None and host_name(host) not in hosts:
            return reject(400, "bad_host", "this Host is not served here (see --allowed-hosts)")
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            # same origin means the same host and port; a page on another port of the same machine is another origin
            netloc = urlsplit(origin).netloc.lower()
            if netloc != host.lower() and host_name(netloc) not in listed:
                return reject(403, "cross_origin", "requests from web pages of another origin are refused")
        return await call_next(request)

    @app.get("/health")
    async def health() -> Response:
        return _json(200, {"ok": True, "model": settings.model_name, "server": "sonda-server",
                           "version": __version__, "validation": validation_state()})

    @app.get("/v1/model")
    async def model_info(request: Request) -> Response:
        if not authorized(request, settings.api_key_env):
            return reject(401, "unauthorized", "missing or wrong API key")
        return _json(200, {"model": settings.model_name, "server": "sonda-server", "version": __version__,
                           "calibration": engine.config.as_dict(), "validation": validation_state(),
                           "settings": settings.public()})

    @app.get("/metrics")
    async def metrics_text(request: Request) -> Response:
        if not settings.public_metrics and not authorized(request, settings.api_key_env):
            return reject(401, "unauthorized", "missing or wrong API key")
        return Response(metrics.render(), media_type="text/plain; version=0.0.4",
                        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    @app.post("/admin/reload")
    async def reload(request: Request) -> Response:
        if not key_from_env(settings.admin_key_env):
            return _json(403, {"error": f"reload over HTTP is off; set {settings.admin_key_env} (or send SIGHUP)"})
        if not authorized(request, settings.admin_key_env):
            return _json(401, {"error": "admin key required"})
        config = await asyncio.get_running_loop().run_in_executor(pool, engine.reload)
        log.info(json.dumps({"event": "reload", "calibration": config.as_dict()}))
        return _json(200, {"ok": True, "calibration": config.as_dict()})

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Response:
        started = time.perf_counter()
        if not authorized(request, settings.api_key_env):
            return reject(401, "unauthorized", "missing or wrong API key")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            return reject(413, "too_large", f"the body is larger than {settings.max_body_bytes} bytes")
        body = await request.body()
        if len(body) > settings.max_body_bytes:
            return reject(413, "too_large", f"the body is larger than {settings.max_body_bytes} bytes")
        try:
            data = orjson.loads(body)
        except orjson.JSONDecodeError:
            return reject(400, "invalid_json", "the body is not valid JSON")
        errors = validate_request(data, settings)
        if errors:
            return reject(422, "invalid", "invalid request", errors)

        loop = asyncio.get_running_loop()
        state, questions = data["state"], data["questions"]
        try:
            lengths = await asyncio.gather(*(loop.run_in_executor(pool, engine.check_lengths, state, q)
                                             for q in questions.values()))
        except Exception:  # noqa: BLE001 - a tokenizer failure is the request's fault, not the server's
            log.exception("length check failed")
            return reject(422, "invalid", "the request could not be tokenized")
        too_long = [{"loc": f"questions.{qid}", "msg": msg} for qid, errs in zip(questions, lengths) for msg in errs]
        if too_long:
            return reject(422, "too_long", "a prompt is too long", too_long)

        tasks = [loop.run_in_executor(pool, engine.decide, state, q) for q in questions.values()]
        try:
            raw = await asyncio.wait_for(asyncio.gather(*tasks), timeout=settings.request_timeout_s)
        except Overloaded:
            return reject(503, "overloaded", "the server is busy; retry shortly", headers={"Retry-After": "1"})
        except (asyncio.TimeoutError, TimeoutError):
            return reject(504, "timeout", f"no answer within {settings.request_timeout_s:g} s")
        except InputTooLong as error:
            return reject(422, "too_long", "a prompt is too long", [{"loc": "questions", "msg": str(error)}])
        except Exception:  # noqa: BLE001 - never return a traceback
            log.exception("decision failed")
            metrics.inc("requests_total", status="500")
            return _json(500, {"error": "internal error"})

        answers, omitted, tokens = {}, [], 0
        for qid, question, result in zip(questions, questions.values(), raw):
            tokens += int(result.get("input_tokens") or 0) if isinstance(result, dict) else 0
            if settings.output_filter:
                kept, dropped = filter_answer(question, result, settings.expose_temperature)
                for name in dropped:
                    if name != "input_tokens":
                        metrics.inc("dropped_fields_total", field=name)
                if kept is None:
                    omitted.append(qid)
                    metrics.inc("omitted_answers_total")
                    continue
                result = kept
            result = dict(result)
            result.pop("input_tokens", None)  # reported once, in usage
            answers[qid] = result
        latency = time.perf_counter() - started
        metrics.inc("questions_total", len(questions))
        metrics.inc("requests_total", status="200")
        metrics.observe("request_seconds", latency)
        payload = {"model": data.get("model") or settings.model_name, "answers": answers,
                   "usage": {"input_tokens": tokens, "output_tokens": 0}, "latency_ms": round(latency * 1e3, 2)}
        if omitted:
            payload["omitted"] = omitted
        return _json(200, payload)

    if settings.demo:
        page = (WEB / "index.html").read_bytes()
        examples = (WEB / "examples.json").read_bytes()

        @app.get("/")
        async def demo_page() -> Response:
            return Response(page, media_type="text/html; charset=utf-8", headers=PAGE_HEADERS)

        @app.get("/demo/examples.json")
        async def demo_examples() -> Response:
            return Response(examples, media_type="application/json",
                            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})

    @app.exception_handler(404)
    async def not_found(request: Request, exc) -> Response:  # noqa: ARG001
        return _json(404, {"error": "not found"})

    app.state.pool = pool  # request threads; idle ones end with the process
    return app
