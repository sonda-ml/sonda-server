# SPDX-License-Identifier: Apache-2.0
"""From a typed question to its answer: tokens, letter logits, temperature, probabilities.

For one question:

1. `options.decision_options` lists (option id, option text);
2. every pass (one, or several for more than 16 options, spread.py) renders the prompt (prompt.py) and
   tokenizes it without special tokens;
3. the pass's token ids go to `read_logits(ids, count)`, which returns the logits of the first `count` answer
   letters at the last position (model.py computes them; batcher.py groups passes of many requests);
4. probabilities: softmax(logits / T), where T is the calibration temperature of the question's type
   (config.ModelConfig.temperature_for);
5. more than 16 options: spread.py combines the passes and sharpens with the knockout temperature;
6. answers.answer builds the /v1/systemone answer.

`check_lengths` runs step 2 for every first-level pass before anything reaches the model, so a request whose
evidence is too long is rejected whole instead of failing half-way.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .answers import answer
from .config import ModelConfig, Settings
from .options import decision_options
from .prompt import LETTERS, prompt_text
from .spread import groups, spread

ReadLogits = Callable[[list[int], int], np.ndarray]


class InputTooLong(ValueError):
    """A pass is longer than the server's max_input_tokens."""


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities of the letters of one pass: softmax(logits / temperature), computed in float64."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


class Engine:
    def __init__(self, tokenizer, read_logits: ReadLogits, config: ModelConfig, settings: Settings,
                 config_dir: str | None = None):
        self.tokenizer = tokenizer
        self.read_logits = read_logits
        self.config = config            # swapped whole by reload(); readers take one reference per question
        self.settings = settings
        self.config_dir = config_dir or settings.model

    def reload(self) -> ModelConfig:
        """Read sonda.conf again (after a new calibration); questions already running keep the old one."""
        self.config = ModelConfig.load(self.config_dir)
        return self.config

    def encode(self, state, criterion, texts: list[str]) -> list[int]:
        return self.tokenizer.encode(prompt_text(state, criterion, texts), add_special_tokens=False)

    def check_lengths(self, state, question: dict) -> list[str]:
        """Errors for first-level passes longer than max_input_tokens (no model call)."""
        texts = [text for _, text in decision_options(question)]
        if len(texts) <= len(LETTERS):
            passes = [texts]
        else:
            passes = [[texts[i] for i in run] for run in groups(len(texts), -(-len(texts) // len(LETTERS)))]
        limit = self.settings.max_input_tokens
        longest = max(len(self.encode(state, question["instructions"], p)) for p in passes)
        if longest > limit:
            return [f"the prompt is {longest} tokens, longer than the limit of {limit}"]
        return []

    def decide(self, state, question: dict) -> dict:
        """The /v1/systemone answer of one question (blocking: waits for its passes)."""
        config = self.config
        temperature = config.temperature_for(question["type"])
        options = decision_options(question)
        tokens = 0

        def read(texts: list[str]) -> list[float]:
            nonlocal tokens
            ids = self.encode(state, question["instructions"], texts)
            if len(ids) > self.settings.max_input_tokens:
                raise InputTooLong(f"a pass is {len(ids)} tokens, longer than {self.settings.max_input_tokens}")
            tokens += len(ids)
            return [float(p) for p in softmax(self.read_logits(ids, len(texts)), temperature)]

        second = config.knockout_temperature if self.settings.method == "knockout" else None
        probs = spread(read, [text for _, text in options], self.settings.method, second)
        result = answer(question, {key: p for (key, _), p in zip(options, probs, strict=True)}, tokens)
        if self.settings.expose_temperature:
            result["temperature"] = temperature
        return result
