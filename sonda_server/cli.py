# SPDX-License-Identifier: Apache-2.0
"""sonda-server command line.

    sonda-server --model /path/to/model --port 8090
    curl -s localhost:8090/v1/systemone -d '{"state": "I was billed twice, please refund",
        "questions": {"refund": {"type": "noul", "instructions": "Asks for money back?"}}}'

Every option also reads an environment variable SONDA_<OPTION> (e.g. SONDA_PORT=8097,
SONDA_INPUT_VALIDATION=0); the command line wins. Keys are read from the environment variables named by
--api-key-env / --admin-key-env, never from the command line. SIGHUP re-reads the model's sonda.conf.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

from . import __version__
from .config import ModelConfig, Settings


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sonda-server", description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="local model folder (weights, tokenizer, sonda.conf)")
    ap.add_argument("--name", help="model name reported in answers (default: the folder name)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--device", help="CUDA device, e.g. cuda or cuda:1")
    ap.add_argument("--method", choices=["knockout", "tree"], help="reading more than 16 options")
    group = ap.add_argument_group("limits")
    for name, kind in (("max-body-bytes", int), ("max-state-chars", int), ("max-questions", int),
                       ("max-options", int), ("max-levels", int), ("max-key-chars", int),
                       ("max-input-tokens", int)):
        group.add_argument(f"--{name}", type=kind)
    group = ap.add_argument_group("batching")
    for name, kind in (("batch-window-ms", float), ("max-batch", int), ("token-budget", int), ("bucket", int),
                       ("max-queue", int), ("request-timeout-s", float), ("threads", int)):
        group.add_argument(f"--{name}", type=kind)
    group = ap.add_argument_group("validation (startup only)")
    group.add_argument("--no-input-validation", dest="input_validation", action="store_false", default=None,
                       help="check only what the prompt needs")
    group.add_argument("--no-output-filter", dest="output_filter", action="store_false", default=None,
                       help="return answers without the per-field filter")
    group.add_argument("--expose-temperature", action="store_true", default=None,
                       help="add the temperature used to every answer")
    ap.add_argument("--no-demo", dest="demo", action="store_false", default=None,
                    help="do not serve the page with API examples and the game at /")
    group = ap.add_argument_group("access")
    group.add_argument("--allowed-hosts", help="comma-separated Host names this server answers to (default: the loopback "
                       "names when listening on loopback, any otherwise; * = any)")
    group.add_argument("--public-metrics", action="store_true", default=None,
                       help="serve /metrics without the API key (only on a private network)")
    group = ap.add_argument_group("keys (names of environment variables)")
    group.add_argument("--api-key-env", help="Bearer key for /v1/systemone (default SONDA_API_KEY; unset = open)")
    group.add_argument("--admin-key-env", help="Bearer key for /admin/reload (default SONDA_ADMIN_KEY)")
    ap.add_argument("--version", action="version", version=f"sonda-server {__version__}")
    return ap


def settings_from_args(argv: list[str] | None = None) -> Settings:
    args = vars(build_parser().parse_args(argv))
    return Settings.from_env(**{key: value for key, value in args.items() if value is not None})


def main(argv: list[str] | None = None) -> int:
    settings = settings_from_args(argv)
    if not settings.model:
        build_parser().error("--model (or SONDA_MODEL) is required: a local model folder")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    log = logging.getLogger("sonda_server")

    import uvicorn  # noqa: PLC0415 - only the running server needs it

    from .batcher import Batcher  # noqa: PLC0415
    from .engine import Engine  # noqa: PLC0415
    from .metrics import Metrics  # noqa: PLC0415
    from .model import LetterModel  # noqa: PLC0415
    from .server import create_app  # noqa: PLC0415

    config = ModelConfig.load(settings.model)
    if config.source.startswith("default"):
        log.warning(f"no sonda.conf in {settings.model}: every question type uses temperature 1.0")
    elif not config.source.endswith("sonda.conf"):
        log.warning(f"calibration read from the older file name {config.source}; rename it to sonda.conf")
    if not settings.input_validation:
        log.warning("input validation is OFF (--no-input-validation): only what the prompt needs is checked")
    if not settings.output_filter:
        log.warning("output filter is OFF (--no-output-filter): answers are returned unfiltered")
    if settings.host not in ("127.0.0.1", "localhost", "::1"):
        if not os.environ.get(settings.api_key_env):
            log.warning(f"listening on {settings.host} without an API key ({settings.api_key_env} is not set): "
                        "anyone who can reach this port can use the model")
        else:
            log.warning("the API key travels in clear text over plain HTTP: put a TLS reverse proxy or an SSH tunnel "
                        "in front when clients are on other machines")

    model = LetterModel(settings.model, device=settings.device, bucket=settings.bucket)
    metrics = Metrics()
    batcher = Batcher(model.letter_logits, settings, metrics)
    engine = Engine(model.tokenizer, batcher.read, config, settings)
    app = create_app(settings, engine, metrics)

    def reload_on_hup(signum, frame) -> None:  # noqa: ARG001
        log.info(json.dumps({"event": "reload", "calibration": engine.reload().as_dict()}))

    signal.signal(signal.SIGHUP, reload_on_hup)
    log.info(json.dumps({"event": "start", "server": "sonda-server", "version": __version__,
                         "model": settings.model_name, "url": f"http://{settings.host}:{settings.port}/v1/systemone",
                         "demo": f"http://{settings.host}:{settings.port}/" if settings.demo else None,
                         "calibration": config.as_dict(),
                         "validation": {"input": settings.input_validation, "output": settings.output_filter}}))
    # The readiness line: scripts that start the server can wait for it.
    print(f"serving {settings.model} on http://{settings.host}:{settings.port}/v1/systemone", flush=True)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning", access_log=False)
    batcher.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
