# SPDX-License-Identifier: Apache-2.0
"""The model on vLLM: the same interface as model.LetterModel (a tokenizer and `letter_logits(batch)`), computed by vLLM.

Selected with `--engine vllm`. Every prompt is sent as its token ids (`TokensPrompt`) with
`SamplingParams(max_tokens=1, logprob_token_ids=<the 16 letter tokens>)`; vLLM returns, for the position after the
prompt, the log-softmax over the whole vocabulary of those 16 tokens. A log-softmax differs from the logits by one
constant per prompt, so softmax(value / T) over the letters equals softmax(logit / T): the engine's temperatures and
answers come out the same as with the transformers model, up to kernel round-off.

vLLM brings its own batching, CUDA graphs and kernels (among them the linear-attention kernels Qwen3.5 needs to be fast)
and a prefix cache. The batcher of sonda-server still groups concurrent prompts; each group is one `generate` call.
vLLM is imported only here, so the default transformers engine needs no vLLM installed.
"""

from __future__ import annotations

import numpy as np

from .prompt import LETTERS


def letter_rows(outputs, slots: list[int]) -> np.ndarray:
    """Letter log-probabilities, shape (len(outputs), len(slots)), from vLLM RequestOutputs."""
    rows = []
    for out in outputs:
        values = out.outputs[0].logprobs[0]  # {token id: Logprob} at the answer position
        rows.append([values[token].logprob for token in slots])
    return np.asarray(rows, dtype=np.float32)


class VllmLetterModel:
    def __init__(self, source: str, max_model_len: int = 32768, gpu_memory_utilization: float = 0.25,
                 extra: dict | None = None):
        import transformers  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415 - only when this engine is chosen
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        # The same tokenizer object as the transformers engine, so prompts become the same token ids.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(source)
        slots = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in slots):
            raise ValueError("every answer letter must be one token of this tokenizer")
        self.slots = [ids[0] for ids in slots]
        self._prompt = TokensPrompt
        args = dict(model=source, tokenizer=source, dtype="bfloat16", max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0,
                    disable_log_stats=True)
        args.update(extra or {})  # --vllm-args, e.g. {"quantization": "fp8"}: FP8 weights, dynamic activation scales
        self.llm = LLM(**args)
        self.params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=self.slots, detokenize=False)

    def letter_logits(self, batch: list[list[int]]) -> np.ndarray:
        """Letter values, shape (len(batch), 16): log-probabilities, equal to the logits up to a constant per row."""
        outputs = self.llm.generate([self._prompt(prompt_token_ids=ids) for ids in batch], self.params, use_tqdm=False)
        return letter_rows(outputs, self.slots)
