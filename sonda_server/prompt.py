# SPDX-License-Identifier: Apache-2.0
"""The decision prompt: what the model reads for one pass.

A pass shows the model the evidence, the criterion (the question's instructions) and up to 16 options,
each behind one capital letter, and asks for the letter alone. The prompt is a two-message chat:

    system   SYSTEM (fixed instruction)
    user     {"evidence": <state>, "criterion": <instructions>,
              "options": [{"letter": "A", "description": "<option text>"}, ...]}

rendered with Qwen3.5's chat template with thinking switched off, so the next token after the assistant
header is the answer letter. The server never generates that token: it reads the logits of the 16 letter
tokens at that position (see engine.py).

Do not change SYSTEM, the payload keys, their order, `ensure_ascii=False`, LETTERS or CHAT_TEMPLATE without
retraining: the served models learnt exactly this text, and any difference moves their answers.
"""

from __future__ import annotations

import json

# One letter per option in a pass; a question with more options is read in several passes (spread.py).
LETTERS = "ABCDEFGHIJKLMNOP"

SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

# Qwen3.5's chat template with thinking off, written out so no template engine runs per request; identical to
# tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False) (tests/test_reference.py).
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def messages(state, criterion, options: list[str]) -> list[dict]:
    """The chat messages of one pass.

    `state` and `criterion` are any JSON value (text, object or list) and are embedded as they are;
    `options` are the option texts of this pass, in order, one per letter (at most 16)."""
    if len(options) > len(LETTERS):
        raise ValueError(f"a pass shows at most {len(LETTERS)} options, got {len(options)}")
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": text} for i, text in enumerate(options)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def prompt_text(state, criterion, options: list[str]) -> str:
    """The full prompt of one pass as text, chat template included; tokenize it without special tokens."""
    system, user = messages(state, criterion, options)
    return CHAT_TEMPLATE.format(system=system["content"], user=user["content"])
