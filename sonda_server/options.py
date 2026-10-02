# SPDX-License-Identifier: Apache-2.0
"""From a typed question to the options the model chooses between.

Every question type becomes a list of (option id, option text):

    noul    ("true", ...), ("false", ...)   the texts are the question's criteria["true"/"false"] when given,
                                            else "The proposition is true." / "... false."
    choice  one option per criteria key, in the given order (a list of texts uses each text as its own key)
    score   ("0", level 0), ("1", level 1), ...   one option per level, ids are the level indices

The text shown to the model is "<id>: <description>", where a description that is an object or a list is
written as compact JSON with sorted keys. The answer probabilities come back keyed
by the option ids, which is how the API reports them.
"""

from __future__ import annotations

import json

QUESTION_TYPES = ("noul", "choice", "score")


def option_pairs(question: dict) -> list[tuple[str, object]]:
    """(option id, raw description) in display order."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        return [(key, (criteria or {}).get(key) or f"The proposition is {key}.") for key in ("true", "false")]
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        return [(key, value or key) for key, value in criteria.items()]
    if kind == "score":
        return [(str(index), level) for index, level in enumerate(criteria)]
    raise ValueError(f"unknown question type {kind!r}")


def option_text(option_id: str, description) -> str:
    """The text the model reads for one option."""
    if isinstance(description, (dict, list)):
        description = json.dumps(description, ensure_ascii=False, sort_keys=True)
    return f"{option_id}: {description}"


def decision_options(question: dict) -> list[tuple[str, str]]:
    """(option id, option text) for every option of a typed question, in display order."""
    return [(key, option_text(key, description)) for key, description in option_pairs(question)]
