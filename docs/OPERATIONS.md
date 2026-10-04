# Running sonda-server

## Environment

sonda-server needs a CUDA GPU, PyTorch with CUDA and transformers (a version that loads Qwen3.5). Install it with
`pip install -e ".[gpu]"` in an environment that has (or may get) a CUDA build of torch, or without the extra
when torch is already there:

```bash
python -m venv .venv
.venv/bin/pip install -e .            # fastapi, uvicorn, orjson, pydantic, numpy
```

To reuse the torch and transformers of another environment instead of installing them again, add its
site-packages after the server's own with a `.pth` file:

```bash
echo "import site; site.addsitedir('/path/to/other/venv/lib/python3.12/site-packages')" \
    > .venv/lib/python3.12/site-packages/zz_shared_site_packages.pth
```

## Start

```bash
sonda-server --model /path/to/model --port 8090
```

The model folder holds the weights, the tokenizer and, optionally, `sonda.conf` with the calibration. The
start log lists the model, the calibration it read and the validation switches; the line
`serving <model> on http://<host>:<port>/v1/systemone` means it is ready, so scripts that start the server can
wait for it. `http://<host>:<port>/` is a page with API examples to try and a small game (see docs/API.md). One process serves one model on one GPU; run another process (another port, `--device cuda:1`) for
another model.

## Options

Every option can also be set as `SONDA_<OPTION>` in the environment (upper case, `-` → `_`); the command line
wins.

| option | default | meaning |
|---|---|---|
| `--model` | required | local model folder: weights, tokenizer, `sonda.conf` |
| `--name` | folder name | model name in answers when the request names none |
| `--host`, `--port` | `127.0.0.1`, `8090` | listen address (keep 127.0.0.1 unless a proxy or key protects it) |
| `--device` | `cuda` | CUDA device |
| `--method` | `knockout` | reading more than 16 options: `knockout` or `tree` |
| `--max-body-bytes` | 2,000,000 | largest request body |
| `--max-state-chars` | 256,000 | largest evidence (as JSON) |
| `--max-questions` | 32 | questions per request |
| `--max-options` | 256 | options of a choice question |
| `--max-levels` | 16 | levels of a score question |
| `--max-key-chars` | 64 | length of an option key |
| `--max-input-tokens` | 32768 | longest prompt of one pass |
| `--batch-window-ms` | 3 | how long the GPU worker waits to fill a batch after the first prompt |
| `--max-batch` | 32 | prompts per forward |
| `--token-budget` | 32768 | rows × padded length per forward |
| `--bucket` | 32 | lengths are padded to a multiple of this |
| `--max-queue` | 1024 | prompts waiting; more → 503 |
| `--request-timeout-s` | 120 | longer → 504 |
| `--threads` | 64 | request threads (tokenizing, many-option logic) |
| `--no-input-validation` | off | check only what the prompt needs |
| `--no-output-filter` | off | return answers without the per-field filter |
| `--expose-temperature` | off | add the temperature used to every answer |
| `--no-demo` | off | do not serve the page with API examples and the game at `/` |
| `--allowed-hosts` | loopback names | Host names served (`*` = any); listening beyond loopback the default is any |
| `--public-metrics` | off | `/metrics` without the API key (a scraper on a private network) |
| `--api-key-env` | `SONDA_API_KEY` | environment variable with the API key; unset or empty = no key needed |
| `--admin-key-env` | `SONDA_ADMIN_KEY` | environment variable with the key for `/admin/reload` |

The validation switches are for measuring their cost, comparing answers with another server and debugging a
client; they are shown in the start log, `/health`, `/v1/model` and the metrics.

## Temperatures and recalibration

The temperature of each question type comes from `sonda.conf` (JSON) in the model folder;

```json
{"temperature": 1.0, "temperature_per_type": {"choice": 1.1, "noul": 0.65, "score": 1.1}}
```

After a new calibration has written `sonda.conf`, load it without a restart: `kill -HUP <pid>`, or `POST /admin/reload` with the admin key. `GET /v1/model` shows what is in use.

## Tuning the batches

* **One client, lowest latency:** `--batch-window-ms 0` (no wait; a forward per prompt as it arrives).
* **Many clients:** the default 3 ms window already merges concurrent prompts; raise `--max-batch` and
  `--token-budget` while GPU memory allows. Watch `sonda_batch_size` and `sonda_queue_wait_seconds` in
  `/metrics`.
* **Long documents:** a 6,000-token prompt alone uses a fifth of the default token budget; long and short
  prompts are sorted into separate forwards automatically.

## Memory

A bf16 4B model needs about 9 GB plus activations for each forward; the activations grow with `--token-budget`
and `--max-batch`. On a GPU whose memory is shared with the rest of the machine (such as the GB10), leave room for
other jobs and lower the token budget if a forward runs out of memory: the affected requests fail, the server
keeps running.

## Logs and metrics

Logs are JSON lines on stderr (start, reload, rejections with their reason); request content is never logged.
`GET /metrics` is Prometheus text (with an API key set, a scraper sends it: `authorization: {credentials: ...}`
in its Prometheus job, or start with `--public-metrics` on a private network); `GET /health` answers without the
API key. The page's Metrics tab shows the same numbers. Access rules: docs/API.md, "Access".
