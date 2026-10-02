# SPDX-License-Identifier: Apache-2.0
"""The answer of one question in TypeSafe's /v1/systemone shape.

From the probability of every option id (options.py) and the input token count:

    noul    {"type": "noul",   "noul": P(true),                       "confidence": max p, "input_tokens": n}
    choice  {"type": "choice", "choice": most likely id, "probabilities": {id: p}, "confidence": max p, ...}
    score   {"type": "score",  "score": sum(level x p), "probabilities": {"0": p0, ...}, "confidence": max p, ...}

`score` is the expected level, not the most likely one. The server moves `input_tokens` from every answer into
the response's `usage`.
"""

from __future__ import annotations


def answer(question: dict, probs: dict[str, float], tokens: int) -> dict:
    """The /v1/systemone answer for a distribution over the question's option ids."""
    kind = question["type"]
    out = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
    if kind == "noul":
        out["noul"] = probs["true"]
    elif kind == "choice":
        out.update(choice=max(probs, key=probs.get), probabilities=probs)
    else:
        out.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
    return out
