# sonda-server architecture

How a request becomes an answer, and why each step is the way it is.

## 1. Request flow

```
client ──HTTP──> uvicorn / FastAPI (one asyncio event loop)
                   │ 1. API key (if SONDA_API_KEY is set)         401
                   │ 2. body size, JSON                            413, 400
                   │ 3. request schema (schemas.py)                422
                   │ 4. prompt length of every first pass          422   (request threads, tokenizer)
                   │ 5. every question in a request thread: engine.decide
                   ▼
           engine.decide ── for each pass: prompt → token ids ── batcher.read ──┐
                                                                                 ▼
                                  GPU worker thread: collects passes of all requests,
                                  sorts by length, cuts into forwards, model.letter_logits
                                                                                 │
           softmax(logits / T_type) ── spread (> 16 options) ── answer ◄────────┘
                   │ 6. answer filter (schemas.filter_answer)      omitted
                   ▼
              200 {"model", "answers", "usage", "latency_ms", "omitted"?}
```

The event loop never computes: it parses, validates and waits. Tokenizing and the spread logic run in a pool of
request threads (`--threads`); the GPU is used by exactly one worker thread (`batcher.py`), so there is one CUDA
stream and no locking around the model.

## 2. The prompt (prompt.py, options.py)

One **pass** shows the model up to 16 options. For the question

```json
{"type": "choice", "instructions": "Which desk handles it?",
 "criteria": {"refunds": "Refunds desk", "sales": "Sales desk"}}
```

and the evidence `"Order 7120 asks for a refund."`, the options become `refunds: Refunds desk` and
`sales: Sales desk` (id, colon, description; an object or list description is written as compact JSON with
sorted keys), and the prompt text is

```
<|im_start|>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>
<|im_start|>user
{"evidence": "Order 7120 asks for a refund.", "criterion": "Which desk handles it?", "options": [{"letter": "A", "description": "refunds: Refunds desk"}, {"letter": "B", "description": "sales: Sales desk"}]}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

tokenized without special tokens. Yes/no questions always have the options `true` and `false` (their texts are
the question's `criteria["true"]` / `["false"]` when given, else "The proposition is true." / "... false.").
Score questions have one option per level, ids `"0"`, `"1"`, ….

This text is the one the models were trained on. It must not change
without retraining. `tests/test_parity_cpu.py` checks, for every test record (all question types and field
shapes, Polish text, quotes, more than 16 options), that it equals the recorded reference prompts and the model
tokenizer's own chat template, token for token.

## 3. The readout (model.py, engine.py)

Nothing is generated. The model body runs over the prompt; the hidden state `h` of the last prompt token is
multiplied by the 16 output-layer rows of the letter tokens A…P:

    logit_k = h · W[letter_k]        k = 1..16 (only the first n, for n options)

which equals the LM head's logits for those tokens without computing the rest of the vocabulary. Then

    p_k = exp(logit_k / T) / Σ_j exp(logit_j / T)

where **T is the calibration temperature of the question's type** (`sonda.conf`:
`temperature_per_type[type]`, else `temperature`, else 1.0). Example: with `temperature_per_type`
noul 0.65, choice 1.1, score 1.1, a yes/no question with letter logits (2.0, 1.0) gives
softmax((2.0, 1.0) / 0.65) = (0.82, 0.18), the same logits for a choice give (0.71, 0.29).

The answer (answers.py): `noul` = P(true); `choice` = the most likely id plus all `probabilities`; `score` = the
expected level Σ level × p plus all `probabilities`; `confidence` = the highest probability.

## 4. More than 16 options (spread.py)

A question with n > 16 options is read in ceil(n / 16) groups of near-equal size plus a final (knockout, the
default) or a group pass plus one pass per group (tree). Knockout with 40 options: groups of 14, 13 and 13 are
read; each keeps its top 5 (16 // 3), the 16th place goes to the next most likely option of any group; a final of
16 is read. A finalist gets its final probability times the probability that the answer is a finalist; any other
option gets its group's share of the final times its probability inside its group. The combination is sharpened
with `knockout_temperature` (the model's, else 0.77): q^(1/T), renormalised. Each of these passes goes through
the batcher like any other.

## 5. Batching (batcher.py, model.py)

The worker waits for the first pass, then collects for `batch_window_ms` (default 3 ms) or until `max_batch`
passes wait. It sorts the passes by length and cuts them into forwards with at most `max_batch` rows and
rows × padded length ≤ `token_budget` (lengths rounded up to `bucket`), so one long document does not pad a
batch of short ones. One forward per group; every pass gets its row.

**Why right padding does not change answers.** Prompts in a batch are padded on the right with token 0. The
model is causal — its attention layers and Qwen3.5's linear-attention (recurrent) layers alike only look at
earlier positions — so the padding after a prompt's last real token cannot change that token's hidden state, the
only one read. Numerically, bf16 kernels may round slightly differently
for different batch shapes; the GPU parity test bounds the difference.

**Back-pressure.** More than `max_queue` waiting passes: the request gets 503 with `Retry-After: 1` at once.
A request that takes longer than `request_timeout_s` gets 504.

## 6. Validation and filtering (schemas.py)

Requests are checked against a closed schema (unknown fields rejected) and limits before any token is computed;
the whole request is rejected with every error listed. Answers are checked field by field after the model;
invalid fields are dropped, an answer without its core field is listed in `omitted`. Both can be switched off at
start (`--no-input-validation`, `--no-output-filter`); what the prompt itself needs, the size and token limits,
and valid JSON output (NaN → null) stay on. See API.md.

## 7. Credits

Parts of the decision runtime (prompt, option mapping, many-option readout, answer shape) are adapted from
third-party work; NOTICE names the sources, their licenses and what was changed.
