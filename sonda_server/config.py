# SPDX-License-Identifier: Apache-2.0
"""Server settings and the model's calibration (temperatures).

Two kinds of configuration:

* **Settings** (`Settings`): how the server runs — address, limits, batching, validation switches, keys. Each
  has a command-line option (cli.py) and an environment variable `SONDA_<NAME>`; the command line wins.
* **Model calibration** (`ModelConfig`): read from `sonda.conf` (JSON) next to the weights, written by whatever
  calibrates the model.

      {"temperature": 1.0,                                   # every type without its own entry
       "temperature_per_type": {"choice": 1.1, "noul": 0.65, "score": 1.1},
       "knockout_temperature": 0.77}                         # optional: more than 16 options

  The question's `type` field selects the temperature; the letter logits of every pass are divided by it
  before the softmax. A missing file means temperature 1.0 (and a warning at start). The file can be read
  again while the server runs (`POST /admin/reload` or SIGHUP), so a new calibration needs no restart.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_FILE = "sonda.conf"
# --typesafe-compat: the requests TypeSafe's API takes that the default validation refuses (see schemas.py)
TYPESAFE_MAX_QUESTIONS = 64


@dataclass(frozen=True)
class ModelConfig:
    """Calibration of one model: the temperature per question type and for the many-option combination."""

    temperature: float = 1.0
    temperature_per_type: dict[str, float] = field(default_factory=dict)
    knockout_temperature: float | None = None
    source: str = "default (no sonda.conf)"

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelConfig":
        """Read sonda.conf from a local model folder; defaults when neither
        exists."""
        path = Path(model_dir) / CONFIG_FILE
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        per_type = {str(k): float(v) for k, v in (data.get("temperature_per_type") or {}).items()}
        for kind, value in [("temperature", data.get("temperature", 1.0)), *per_type.items()]:
            if not float(value) > 0:
                raise ValueError(f"{path}: temperature {kind} must be positive, got {value}")
        knockout = data.get("knockout_temperature")
        return cls(temperature=float(data.get("temperature", 1.0)), temperature_per_type=per_type,
                   knockout_temperature=float(knockout) if knockout is not None else None, source=str(path))

    def temperature_for(self, question_type: str) -> float:
        """The temperature of this question type, else the model's single temperature."""
        return self.temperature_per_type.get(question_type, self.temperature)

    def as_dict(self) -> dict:
        return asdict(self)


def _env_bool(value: str) -> bool:
    return value.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class Settings:
    """Everything the server needs besides the model's own files. Defaults suit one GPU."""

    model: str = ""                     # local model folder (required)
    name: str | None = None              # reported as "model" when a request names none; default: folder name
    host: str = "127.0.0.1"
    port: int = 8090
    device: str = "cuda"
    method: str = "knockout"             # more than 16 options: knockout or tree (spread.py)
    engine: str = "transformers"         # transformers (model.py) or vllm (vllm_model.py)
    gpu_memory_utilization: float = 0.25  # vllm only: share of the GPU memory vLLM may take (weights + cache)
    vllm_args: str = ""                  # vllm only: JSON object of further vLLM engine arguments, e.g. {"quantization": "fp8"}

    # limits (input validation)
    max_body_bytes: int = 2_000_000
    max_state_chars: int = 256_000
    max_questions: int = 32
    max_options: int = 256
    max_levels: int = 16
    max_key_chars: int = 64
    max_input_tokens: int = 32768        # one pass, after the chat template

    # batching
    batch_window_ms: float = 3.0         # how long the GPU worker waits to fill a batch after the first pass
    max_batch: int = 32                  # passes in one forward
    token_budget: int = 32768            # rows x padded length of one forward
    bucket: int = 32                     # pad lengths to a multiple of this (fewer shapes)
    max_queue: int = 1024                # passes waiting; more -> 503
    request_timeout_s: float = 120.0
    threads: int = 64                    # request threads that tokenize and run `spread`

    # validation switches (startup only)
    input_validation: bool = True
    output_filter: bool = True
    expose_temperature: bool = False
    typesafe_compat: bool = False        # accept what TypeSafe's API accepts: empty evidence, any question id, 64 questions

    # the page with API examples and the game at GET / (web/index.html)
    demo: bool = True

    # access: Host names served ("" = 127.0.0.1/localhost/::1 when listening on loopback, any otherwise; "*" = any)
    allowed_hosts: str = ""
    public_metrics: bool = False         # GET /metrics without the API key (for a scraper on a private network)

    # keys: names of environment variables, never the keys themselves
    api_key_env: str = "SONDA_API_KEY"
    admin_key_env: str = "SONDA_ADMIN_KEY"

    @property
    def model_name(self) -> str:
        return self.name or Path(self.model).name

    @property
    def question_limit(self) -> int:
        """Questions per request: max_questions, at least TYPESAFE_MAX_QUESTIONS with --typesafe-compat."""
        return max(self.max_questions, TYPESAFE_MAX_QUESTIONS) if self.typesafe_compat else self.max_questions

    @classmethod
    def from_env(cls, environ: dict | None = None, **overrides) -> "Settings":
        """Defaults, then SONDA_<FIELD> environment variables, then explicit overrides (the command line)."""
        environ = os.environ if environ is None else environ
        values = {}
        for f in fields(cls):
            raw = environ.get(f"SONDA_{f.name.upper()}")
            if raw is None:
                continue
            kind = type(f.default) if f.default is not None else str
            values[f.name] = _env_bool(raw) if kind is bool else kind(raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def public(self) -> dict:
        """The settings shown by GET /v1/model (no key names' values, which are never stored here anyway)."""
        out = asdict(self)
        out["model_name"] = self.model_name
        return out
