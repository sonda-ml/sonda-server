# SPDX-License-Identifier: Apache-2.0
"""Counters and histograms for GET /metrics, in the Prometheus text format (no extra dependency).

Names (all prefixed `sonda_`):

    requests_total{status}              finished /v1/systemone requests by HTTP status
    rejected_total{reason}              rejected requests: invalid_json, too_large, invalid, too_long,
                                        unauthorized, overloaded, timeout, bad_host, cross_origin
    questions_total                     answered questions
    omitted_answers_total               answers left out by the answer filter
    dropped_fields_total{field}         answer fields dropped by the answer filter
    passes_total                        model passes (prompts) computed
    batches_total                       forward passes on the GPU
    batch_size                          histogram: passes per forward
    queue_wait_seconds                  histogram: time a pass waited for its forward
    forward_seconds                     histogram: time of one forward
    request_seconds                     histogram: time of one request
    queue_depth                         gauge: passes waiting now
"""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = {
    "batch_size": (1, 2, 4, 8, 16, 32, 64),
    "queue_wait_seconds": (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    "forward_seconds": (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    "request_seconds": (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
}


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.histograms: dict[str, list] = {name: [[0] * (len(b) + 1), 0.0, 0] for name, b in BUCKETS.items()}
        self.gauges: dict[str, float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float) -> None:
        bounds = BUCKETS[name]
        with self._lock:
            counts, _, _ = hist = self.histograms[name]
            index = next((i for i, b in enumerate(bounds) if value <= b), len(bounds))
            counts[index] += 1
            hist[1] += value
            hist[2] += 1

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def get(self, name: str, **labels) -> float:
        with self._lock:
            return self.counters.get((name, tuple(sorted(labels.items()))), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), value in sorted(self.counters.items()):
                tag = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"sonda_{name}{tag} {value:g}")
            for name, value in sorted(self.gauges.items()):
                lines.append(f"sonda_{name} {value:g}")
            for name, (counts, total, n) in self.histograms.items():
                running = 0
                for bound, count in zip((*BUCKETS[name], "+Inf"), counts):
                    running += count
                    lines.append(f'sonda_{name}_bucket{{le="{bound}"}} {running}')
                lines.append(f"sonda_{name}_sum {total:g}")
                lines.append(f"sonda_{name}_count {n}")
        return "\n".join(lines) + "\n"
