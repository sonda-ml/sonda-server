# SPDX-License-Identifier: Apache-2.0
"""The model on the GPU: one forward pass for a batch of prompts, returning the answer letters' logits.

Loading: the tokenizer and the causal LM from a local folder in bf16 on one CUDA device. Qwen3.5
checkpoints are loaded as `Qwen3_5ForCausalLM` from their text config. Every answer letter A..P must be a single
token; the 16 rows of the output layer (`lm_head`) for those tokens are kept as `slot_weight`.

Readout: run the transformer body (not the full LM head) and take the hidden state of each prompt's last real
token; multiplying it by `slot_weight` gives the 16 letter logits — the same numbers the LM head would give for
those tokens, without computing the other ~150k vocabulary logits.

Batching: prompts of different lengths are right-padded with token 0 to one length (rounded up to `bucket`).
The model is causal (attention and Qwen3.5's linear-attention layers alike), so tokens after a prompt's last
real token cannot change that token's hidden state; only that position is read.
"""

from __future__ import annotations

import numpy as np

from .prompt import LETTERS


class LetterModel:
    def __init__(self, source: str, device: str = "cuda", bucket: int = 32):
        import torch  # noqa: PLC0415 - CUDA libraries load only when a model is served
        import transformers  # noqa: PLC0415

        self.torch = torch
        self.device = device
        self.bucket = max(1, bucket)
        config = transformers.AutoConfig.from_pretrained(source)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        cls = transformers.AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            cls, config = transformers.Qwen3_5ForCausalLM, config.get_text_config()
        self.model = cls.from_pretrained(source, config=config, dtype=torch.bfloat16,
                                         device_map={"": device}).eval()
        slots = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("every answer letter must be one token of this tokenizer")
        self.slots = [ids[0] for ids in slots]
        self.slot_weight = self.model.lm_head.weight[self.slots].detach().contiguous()

    def letter_logits(self, batch: list[list[int]]) -> np.ndarray:
        """Letter logits, shape (len(batch), 16), float32, for prompts given as token ids."""
        torch = self.torch
        lengths = [len(ids) for ids in batch]
        width = -(-max(lengths) // self.bucket) * self.bucket
        ids = torch.zeros((len(batch), width), dtype=torch.long)
        for row, prompt in enumerate(batch):
            ids[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        last = torch.tensor([n - 1 for n in lengths], device=self.device)
        with torch.inference_mode():
            hidden = self.model.model(input_ids=ids.to(self.device), use_cache=False).last_hidden_state
            logits = hidden[torch.arange(len(batch), device=self.device), last] @ self.slot_weight.T
        return logits.float().cpu().numpy()
