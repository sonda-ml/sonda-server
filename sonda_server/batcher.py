# SPDX-License-Identifier: Apache-2.0
"""Dynamic batching: passes from many requests share one forward on the GPU.

Request threads call `read(ids, count)`; it queues the pass and blocks on a Future. One worker thread owns the
GPU:

1. it waits for the first queued pass, then keeps collecting for `batch_window_ms` (or until `max_batch`
   passes wait) so concurrent requests land in the same forward;
2. it sorts what it took by length and cuts it into forwards that respect `max_batch` and `token_budget`
   (rows x padded length, padded to a multiple of `bucket`), so a long document does not pad a batch of short
   ones;
3. one `run_batch` call per forward (model.LetterModel.letter_logits), then every Future gets its row of letter
   logits — or the forward's exception.

Back-pressure: more than `max_queue` waiting passes raise `Overloaded` at once (the server answers 503 with
Retry-After) instead of letting latency grow without bound. A single client sees almost no extra latency: when
nothing else is queued, the window is the only wait.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .metrics import Metrics


class Overloaded(RuntimeError):
    """Too many passes are waiting; try again later."""


@dataclass
class _Pass:
    ids: list[int]
    future: Future = field(default_factory=Future)
    queued: float = field(default_factory=time.perf_counter)


def plan_forwards(lengths: list[int], max_batch: int, token_budget: int, bucket: int) -> list[list[int]]:
    """Indices grouped into forwards: sorted by length, each forward within max_batch rows and
    rows x padded length <= token_budget (a pass longer than the budget gets a forward of its own)."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    forwards, current = [], []
    for index in order:
        padded = -(-lengths[index] // bucket) * bucket
        if current and (len(current) == max_batch or (len(current) + 1) * padded > token_budget):
            forwards.append(current)
            current = []
        current.append(index)
    if current:
        forwards.append(current)
    return forwards


class Batcher:
    def __init__(self, run_batch: Callable[[list[list[int]]], np.ndarray], settings: Settings,
                 metrics: Metrics | None = None):
        self.run_batch = run_batch
        self.settings = settings
        self.metrics = metrics or Metrics()
        self._queue: deque[_Pass] = deque()
        self._cv = threading.Condition()
        self._stopped = False
        self._worker = threading.Thread(target=self._loop, name="sonda-gpu", daemon=True)
        self._worker.start()

    # -- request side --------------------------------------------------------------------------

    def submit(self, ids: list[int]) -> Future:
        item = _Pass(list(ids))
        with self._cv:
            if self._stopped:
                raise RuntimeError("the batcher is stopped")
            if len(self._queue) >= self.settings.max_queue:
                raise Overloaded(f"{len(self._queue)} passes are waiting")
            self._queue.append(item)
            self.metrics.set("queue_depth", len(self._queue))
            self._cv.notify()
        return item.future

    def read(self, ids: list[int], count: int) -> np.ndarray:
        """The first `count` letter logits of one pass (blocks until its forward ran)."""
        return self.submit(ids).result(timeout=self.settings.request_timeout_s)[:count]

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        self._worker.join(timeout=5)

    # -- GPU side --------------------------------------------------------------------------------

    def _take(self) -> list[_Pass]:
        window = self.settings.batch_window_ms / 1000
        with self._cv:
            while not self._queue and not self._stopped:
                self._cv.wait()
            if self._stopped:
                return []
            deadline = time.perf_counter() + window
            while len(self._queue) < self.settings.max_batch and (left := deadline - time.perf_counter()) > 0:
                self._cv.wait(left)
            taken = list(self._queue)
            self._queue.clear()
            self.metrics.set("queue_depth", 0)
        return taken

    def _loop(self) -> None:
        while True:
            taken = self._take()
            if not taken:
                if self._stopped:
                    return
                continue
            plan = plan_forwards([len(p.ids) for p in taken], self.settings.max_batch,
                                 self.settings.token_budget, self.settings.bucket)
            for rows in plan:
                items = [taken[i] for i in rows]
                started = time.perf_counter()
                for item in items:
                    self.metrics.observe("queue_wait_seconds", started - item.queued)
                try:
                    logits = self.run_batch([item.ids for item in items])
                except Exception as error:  # noqa: BLE001 - every waiting request gets the error
                    for item in items:
                        item.future.set_exception(error)
                    continue
                self.metrics.observe("forward_seconds", time.perf_counter() - started)
                self.metrics.observe("batch_size", len(items))
                self.metrics.inc("batches_total")
                self.metrics.inc("passes_total", len(items))
                for item, row in zip(items, logits):
                    item.future.set_result(np.asarray(row))
