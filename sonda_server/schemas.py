# SPDX-License-Identifier: Apache-2.0
"""Request validation (reject what is wrong) and answer filtering (pass what is right, drop the rest).

Request schema — POST /v1/systemone
-------------------------------------------------

    {"state": <text, object or list: the evidence>,
     "questions": {"<id>": <question>, ...},          1..max_questions; id: 1-64 chars of [A-Za-z0-9_.-]
     "model": "<name>"}                                optional, echoed back

    question = {"type": "noul",   "instructions": <text|object|list>, "criteria": {"true": .., "false": ..}?}
             | {"type": "choice", "instructions": .., "criteria": {"<key>": <description>, ...} | ["<text>", ...]}
             | {"type": "score",  "instructions": .., "criteria": [<level 0>, <level 1>, ...]}

Unknown fields are rejected at every level. A request with any error is rejected whole (HTTP 422) and lists
every error as {"loc": "questions.q1.criteria", "msg": "..."}; the model never sees it. The prompt is built from
the client's own JSON (not a re-serialised copy), so key order and values reach the model exactly as sent.

With input validation switched off (`--no-input-validation`) only what the prompt needs is checked: state and
questions present, a known type, instructions present, at least two options.

Answer filter
----------------------------

Every field of an answer is checked on its own and kept only when valid; an invalid field is dropped without an
error. An answer without its core field (`noul` for yes/no, `probabilities` for choice and score) is left out of
`answers` and its id listed in `omitted`. See `filter_answer`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from .config import Settings
from .options import QUESTION_TYPES, option_pairs

QUESTION_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
# Answer fields the API may return; `temperature` only with --expose-temperature.
ANSWER_FIELDS = ("type", "noul", "choice", "score", "probabilities", "confidence", "input_tokens")
SUM_TOLERANCE = 1e-3
MATCH_TOLERANCE = 1e-6


# --------------------------------------------------------------------------------------- request schema

class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


Instructions = Union[str, dict[str, JsonValue], list[JsonValue]]


class NoulCriteria(_Closed):
    true: JsonValue = None
    false: JsonValue = None


class NoulQuestion(_Closed):
    type: Literal["noul"]
    instructions: Instructions
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Closed):
    type: Literal["choice"]
    instructions: Instructions
    criteria: Union[dict[str, JsonValue], list[str]]


class ScoreQuestion(_Closed):
    type: Literal["score"]
    instructions: Instructions
    criteria: list[JsonValue]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class Request(_Closed):
    state: JsonValue
    questions: dict[str, Question]
    model: str | None = None


def _loc(parts) -> str:
    # Drop pydantic's union tags ("choice", "function-after[...]") so the path names the client's fields.
    return ".".join(str(p) for p in parts if not (isinstance(p, str) and (p in QUESTION_TYPES or "[" in p)))


def _empty(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (dict, list)) and not value)


def validate_request(data, settings: Settings) -> list[dict]:
    """Every error of a request body (already parsed JSON); an empty list means valid."""
    if not settings.input_validation:
        return minimal_errors(data)
    try:
        Request.model_validate(data)
    except ValidationError as error:
        return [{"loc": _loc(e["loc"]), "msg": e["msg"]} for e in error.errors(include_url=False)]
    errors = []
    state = data["state"]
    if _empty(state):
        errors.append({"loc": "state", "msg": "the evidence must not be empty"})
    elif len(json.dumps(state, ensure_ascii=False)) > settings.max_state_chars:
        errors.append({"loc": "state", "msg": f"the evidence is longer than {settings.max_state_chars} characters"})
    questions = data["questions"]
    if not 1 <= len(questions) <= settings.max_questions:
        errors.append({"loc": "questions", "msg": f"give 1 to {settings.max_questions} questions"})
    for qid, question in questions.items():
        where = f"questions.{qid}"
        if not QUESTION_ID.fullmatch(qid):
            errors.append({"loc": where, "msg": "a question id is 1-64 characters of letters, digits, _ . -"})
        if _empty(question["instructions"]):
            errors.append({"loc": f"{where}.instructions", "msg": "the instructions must not be empty"})
        criteria = question.get("criteria")
        if question["type"] == "choice":
            keys = list(criteria) if isinstance(criteria, dict) else criteria
            if not 2 <= len(keys) <= settings.max_options:
                errors.append({"loc": f"{where}.criteria", "msg": f"a choice has 2 to {settings.max_options} options"})
            if isinstance(criteria, list) and len(set(criteria)) != len(criteria):
                errors.append({"loc": f"{where}.criteria", "msg": "options must be distinct"})
            for key in keys:
                if not key.strip() or len(key) > settings.max_key_chars:
                    errors.append({"loc": f"{where}.criteria", "msg":
                                   f"option key {key[:80]!r} must be 1-{settings.max_key_chars} characters, not blank"})
        elif question["type"] == "score":
            if not 2 <= len(criteria) <= settings.max_levels:
                errors.append({"loc": f"{where}.criteria", "msg": f"a score has 2 to {settings.max_levels} levels"})
            if any(_empty(level) for level in criteria):
                errors.append({"loc": f"{where}.criteria", "msg": "a level must not be empty"})
    return errors


def minimal_errors(data) -> list[dict]:
    """What the prompt cannot do without (used with --no-input-validation)."""
    if not isinstance(data, dict):
        return [{"loc": "", "msg": "the body must be a JSON object"}]
    errors = []
    if "state" not in data:
        errors.append({"loc": "state", "msg": "field required"})
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        return errors + [{"loc": "questions", "msg": "an object with at least one question is required"}]
    for qid, question in questions.items():
        where = f"questions.{qid}"
        if not isinstance(question, dict):
            errors.append({"loc": where, "msg": "a question must be an object"})
            continue
        if question.get("type") not in QUESTION_TYPES:
            errors.append({"loc": f"{where}.type", "msg": f"unknown question type {question.get('type')!r}"})
            continue
        if "instructions" not in question:
            errors.append({"loc": f"{where}.instructions", "msg": "field required"})
        criteria = question.get("criteria")
        if question["type"] == "choice" and not (isinstance(criteria, (dict, list)) and len(criteria) >= 2):
            errors.append({"loc": f"{where}.criteria", "msg": "choice criteria must name at least two options"})
        if question["type"] == "score" and not (isinstance(criteria, list) and len(criteria) >= 2):
            errors.append({"loc": f"{where}.criteria", "msg": "score criteria must list at least two levels"})
        if question["type"] == "noul" and criteria is not None and not isinstance(criteria, dict):
            errors.append({"loc": f"{where}.criteria", "msg": "yes/no criteria must be an object"})
    return errors


# ----------------------------------------------------------------------------------------- answer filter

def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(value) -> bool:
    return _number(value) and 0.0 <= value <= 1.0


def filter_answer(question: dict, answer: dict, expose_temperature: bool = False) -> tuple[dict | None, list[str]]:
    """(the valid fields of an answer, the names of the dropped fields); None when its core field is invalid.

    type          equal to the question's type
    noul          a probability (finite, in [0, 1])                                  core field of yes/no
    probabilities exactly the question's option ids, each a probability, summing
                  to 1 within 1e-3                                                   core field of choice/score
    choice        an option id with the highest probability (ties allowed)
    score         finite, between 0 and the last level, equal to sum(level x p)
    confidence    a probability equal to the highest probability of the answer
    input_tokens  an integer >= 0
    temperature   a positive number (only with expose_temperature)
    """
    kind = question.get("type")
    if not isinstance(answer, dict):
        return None, ["<answer>"]
    ids = [key for key, _ in option_pairs(question)]
    out: dict = {}
    if answer.get("type") == kind:
        out["type"] = kind

    probs = answer.get("probabilities")
    if kind in ("choice", "score") and isinstance(probs, dict) and list(probs) and set(probs) == set(ids) \
            and all(_probability(v) for v in probs.values()) and abs(sum(probs.values()) - 1.0) <= SUM_TOLERANCE:
        out["probabilities"] = {key: float(probs[key]) for key in ids}
    if kind == "noul" and _probability(answer.get("noul")):
        out["noul"] = float(answer["noul"])

    kept = out.get("probabilities")
    choice = answer.get("choice")
    if kind == "choice" and choice in ids and kept and kept[choice] >= max(kept.values()) - MATCH_TOLERANCE:
        out["choice"] = choice
    score = answer.get("score")
    if kind == "score" and _number(score) and 0 <= score <= len(ids) - 1 and kept and \
            abs(score - sum(int(k) * v for k, v in kept.items())) <= MATCH_TOLERANCE:
        out["score"] = float(score)

    confidence = answer.get("confidence")
    top = (max(kept.values()) if kept else
           max(out["noul"], 1 - out["noul"]) if "noul" in out else None)
    if _probability(confidence) and top is not None and abs(confidence - top) <= MATCH_TOLERANCE:
        out["confidence"] = float(confidence)
    tokens = answer.get("input_tokens")
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
        out["input_tokens"] = tokens
    temperature = answer.get("temperature")
    if expose_temperature and _number(temperature) and temperature > 0:
        out["temperature"] = float(temperature)

    dropped = [key for key in answer if key not in out]
    core = "noul" if kind == "noul" else "probabilities"
    if "type" not in out or core not in out:
        return None, dropped or [core]
    return out, dropped
