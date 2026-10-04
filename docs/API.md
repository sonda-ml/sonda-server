# sonda-server API

All bodies are JSON (`Content-Type: application/json`). Errors have the form
`{"error": "<message>", "errors": [{"loc": "<path>", "msg": "<message>"}]?}`; no traceback is ever returned.

## POST /v1/systemone

The TypeSafe-style decision call: one piece of evidence, one or more typed questions.

### Request

```json
{
  "state": "Order 7120 was delivered 12 days ago; the customer asks for a refund.",
  "questions": {
    "refund":  {"type": "noul",   "instructions": "Is a refund allowed under the 30-day policy?"},
    "desk":    {"type": "choice", "instructions": "Which desk handles it?",
                "criteria": {"refunds": "Refunds desk", "sales": "Sales desk", "tech": "Tech support"}},
    "urgency": {"type": "score",  "instructions": "How urgent is the reply?",
                "criteria": ["No time limit", "Needed by a stated date", "Needed today"]}
  },
  "model": "my-model"
}
```

| field | rule (defaults; see OPERATIONS.md for the options) |
|---|---|
| `state` | required; text, object or list; not empty; at most `max_state_chars` (256,000) characters as JSON |
| `questions` | required object of 1 to `max_questions` (32) questions; an id is 1-64 characters of `A-Za-z0-9_.-` |
| `model` | optional text, echoed in the response (the server serves one model) |
| `type` | required: `noul` (yes/no), `choice` or `score` |
| `instructions` | required, not empty: text, object or list |
| `criteria` (choice) | `{"<key>": <description>, ...}` or `["<text>", ...]`; 2 to `max_options` (256) options; keys 1-`max_key_chars` (64) characters, not blank, distinct |
| `criteria` (score) | list of 2 to `max_levels` (16) non-empty levels, lowest first |
| `criteria` (noul) | optional `{"true": <text>, "false": <text>}` |
| anything else | rejected (unknown field) |

Each question is read as one prompt (up to 16 options) or several (more options); a prompt longer than
`max_input_tokens` (32768) rejects the request (422) before anything is computed.

### Response 200

```json
{
  "model": "my-model",
  "answers": {
    "refund":  {"type": "noul", "noul": 0.91, "confidence": 0.91},
    "desk":    {"type": "choice", "choice": "refunds", "confidence": 0.88,
                "probabilities": {"refunds": 0.88, "sales": 0.07, "tech": 0.05}},
    "urgency": {"type": "score", "score": 0.62, "confidence": 0.52,
                "probabilities": {"0": 0.43, "1": 0.52, "2": 0.05}}
  },
  "usage": {"input_tokens": 512, "output_tokens": 0},
  "latency_ms": 41.3
}
```

| answer field | meaning |
|---|---|
| `noul` | probability that the answer is true (yes/no only) |
| `choice` | the most likely option key (choice only) |
| `score` | the expected level Σ level × p (score only), not the most likely level |
| `probabilities` | probability of every option key / level index (choice, score) |
| `confidence` | the highest probability of the answer |
| `temperature` | the calibration temperature used — only when the server runs with `--expose-temperature` |

Every field passes the answer filter: it is present only when valid (finite, in range, consistent with the
others). An answer whose core field (`noul`, or `probabilities`) is not valid is left out of `answers`, and its id
is listed in `"omitted": [...]`; the status is still 200. With `--no-output-filter` answers are returned as
computed (non-finite numbers become `null`).

`usage.input_tokens` is the sum over all prompts of all questions; answers do not carry it.

### Errors

| status | when |
|---|---|
| 400 | the body is not JSON |
| 401 | `SONDA_API_KEY` (or the variable named by `--api-key-env`) is set and the request has no matching `Authorization: Bearer <key>` |
| 413 | the body is larger than `max_body_bytes` (2,000,000) |
| 422 | the request breaks the schema or a limit, or a prompt is longer than `max_input_tokens`; `errors` lists every problem with its location |
| 503 | more than `max_queue` prompts are waiting; `Retry-After: 1` |
| 504 | no answer within `request_timeout_s` (120 s) |
| 500 | unexpected server error (logged) |

## GET /health

`{"ok": true, "model": "<name>", "server": "sonda-server", "version": "0.1.0",
"validation": {"input": true, "output": true}}` — open even when an API key is set.

## GET /v1/model

The model name, its calibration (`temperature`, `temperature_per_type`, `knockout_temperature`, the file it
came from), the validation switches and all server settings.

## GET /metrics

Prometheus text format: requests by status, rejections by reason, answered questions, omitted answers, dropped
answer fields, prompts and forwards computed, histograms of batch size, queue wait, forward time and request
time, and the current queue depth. See `sonda_server/metrics.py` for the names.

## GET / and GET /demo/examples.json

A page for people: invented example requests in English and Polish to edit and send to this server, filtered by
question type (`choice`, `noul`, `score`, several at once), kind of data in `state` (text, JSON object, list or
table) and domain; the answers drawn as probability bars, the raw response, and the same request as `curl`,
Python and JavaScript code. Its "Who judges better?" game plays ten cases with a known answer: the player picks an
answer and a confidence, the model answers with probabilities, and each side scores 100 × (1 − Brier / 2) per
round. Snake, Tetris and Pac-Man are played by you or by the model. Steered by the model, the page computes what every
move would do — for Snake whether each direction is safe, how far the food would be and how many free cells would
be left (or, in the "board only" mode, just the board as rows of text); for Tetris every placement with its cleared
lines, new holes, height and bumpiness, of which the 5 best by a classic heuristic are offered in random order; for Pac-Man, per way, the path distance
to the nearest ghost, how many cells Pac-Man would reach before any ghost, what it eats and how far the nearest dot
is (ghosts move a step per turn, so the game waits for the model) —
and asks the model one `choice` question per move. The page shows the model's last decision, the request it sent,
and the model's answer time; the game keeps an even pace a little slower than that time, and a speed slider can
only slow it down further (100% is the fastest pace the model keeps up with). The Metrics tab draws `/metrics`.

The page is self-contained (no external scripts, fonts or CDNs; a strict Content-Security-Policy) and needs no
key; its requests go to `/v1/systemone` like any client's, so with an API key set the page asks for it and keeps
it in the browser tab only. `/demo/examples.json` is the example set the page uses (both languages share option
keys; `expected` holds the answers the game checks; `note` is shown under an example). Examples that would need
arithmetic carry the computed value next to the raw data (a total, a per-person amount, a date): the model
answers in one pass and compares and applies rules well but does not calculate — "Limit of the model: adding up"
shows the same case without the total. `python bench/check_examples.py --url …` asks a running server every
example with a known answer and flags wrong or unsure ones. `--no-demo` (or `SONDA_DEMO=0`) turns both off (404). The
files are read at start: after editing `sonda_server/web/`, restart the server.

## Access

With no API key set (`SONDA_API_KEY` empty or unset, the default) every endpoint is open. With a key set:

| endpoint | without the key |
|---|---|
| `POST /v1/systemone` | 401 |
| `GET /v1/model` | 401 |
| `GET /metrics` | 401, unless `--public-metrics` (for a scraper on a private network) |
| `GET /health`, `GET /`, `GET /demo/examples.json` | open: nothing secret in them |

The page asks for the key after the first 401 and keeps it in memory only (not in storage or cookies); its code
samples use `$SONDA_API_KEY`, never the key. Always, key or not:

* **Host allow-list** (`--allowed-hosts`): a request must name a Host the server serves; listening on loopback
  (the default), only `127.0.0.1`, `localhost` and `::1`. This stops DNS rebinding — a foreign domain pointed at
  127.0.0.1 so that a web page could read a local server. Behind a reverse proxy, list its public name.
* **Other web pages are refused** (403): a POST whose `Origin` is not this server's own (another site, or another
  port of the same machine) — so a page elsewhere cannot use a local server through the visitor's browser.
  Clients outside a browser (curl, Python) send no `Origin` and are not affected.

The key travels as a Bearer header: over plain HTTP between machines anyone on the path can read it, so put a TLS
reverse proxy or an SSH tunnel in front (the start log warns when the server listens beyond loopback). Use a long
random key (`openssl rand -hex 32`); there is no lock-out after wrong keys.

## POST /admin/reload

Re-reads `sonda.conf` (after a new calibration) without a restart; answers already being computed keep
the old temperatures. Needs `Authorization: Bearer <key>` with the key from `SONDA_ADMIN_KEY` (or the variable
named by `--admin-key-env`); without such a variable the endpoint is off (403) and `kill -HUP <pid>` does the
same from the machine itself.
