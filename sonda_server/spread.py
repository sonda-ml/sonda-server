# SPDX-License-Identifier: Apache-2.0
"""Questions with more options than letters (more than 16).

One pass shows at most 16 lettered options. `spread(read, texts)` returns a probability for every option of a
longer list from a reader `read(texts) -> probabilities` that answers one pass of at most 16 options:

* up to 16 options: one pass, `read(texts)` itself, untouched;
* more: the options are split, in their given order, into ceil(n / 16) groups of near-equal size and every
  option is scored (nothing is dropped unread), by one of two methods:

  - "knockout" (default): every group is read; then a final of up to 16 is read with the top
    16 // groups options of each group (at least one), the free places going to the next most likely options
    of any group. Finalists keep the final's probability times the chance that the answer is a finalist; every
    other option gets its group's share of the final times its probability inside the group.
  - "tree": one pass whose letters stand for whole groups ("One of: a; b; c"), then one pass per group:
    P(option) = P(its group) x P(option | its group).

  The combined distribution is then sharpened by a temperature: q ** (1 / T), renormalised. The letter
  temperature is fitted on questions of up to 16 options and leaves the combination underconfident; the default
  is T = 0.77 for the knockout (TEMPERATURES), and a model may carry its own `knockout_temperature`.

Worked example, knockout with 40 options: groups of 14, 13, 13 options (three passes), each keeps its top
16 // 3 = 5, the sixteenth place goes to the next most likely option overall, and one final pass of 16 sets
the finalists' shares: four passes in all. Up to 256 options this takes ceil(n / 16) + 1 passes; beyond, the
final itself is spread recursively.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .prompt import LETTERS

METHODS = ("knockout", "tree")
# Default temperature of the combined distribution over more than 16 options, per method.
TEMPERATURES = {"knockout": 0.77, "tree": 1.0}

Reader = Callable[[list[str]], Sequence[float]]


def groups(n: int, count: int) -> list[range]:
    """`count` contiguous runs covering range(n), their sizes differing by at most one (larger first)."""
    base, extra = divmod(n, count)
    runs, start = [], 0
    for index in range(count):
        stop = start + base + (index < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def spread(read: Reader, texts: list[str], method: str = "knockout",
           temperature: float | None = None) -> list[float]:
    """A probability for every option text, from a reader of at most 16 options per pass (see module doc)."""
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probs = _combine(read, texts, method)
    temperature = TEMPERATURES[method] if temperature is None else temperature
    if temperature != 1.0:
        probs = [q ** (1 / temperature) for q in probs]
        total = sum(probs)
        probs = [q / total for q in probs]
    return probs


def _combine(read: Reader, texts: list[str], method: str) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    if method == "knockout":
        weights = _knockout(read, texts)
    elif method == "tree":
        weights = _tree(read, texts)
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    total = sum(weights)
    return [w / total for w in weights]


def _knockout(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[i] for i in run])) for run in runs]
    inner = [[q / sum(p) for q in p] for p in inner]
    keep = max(1, len(LETTERS) // len(runs))
    # Ties go to the earlier option: the sorts are stable and walk the options in order.
    ranked = [sorted(range(len(p)), key=lambda j, p=p: -p[j]) for p in inner]
    chosen = {(g, j) for g, order in enumerate(ranked) for j in order[:keep]}
    rest = sorted(
        ((g, j) for g, order in enumerate(ranked) for j in order[keep:]),
        key=lambda gj: -inner[gj[0]][gj[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [sorted(j for h, j in chosen if h == g) for g in range(len(runs))]
    final = _combine(read, [texts[run[j]] for run, top in zip(runs, tops) for j in top], "knockout")
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at: at + len(top)])))
        at += len(top)
    in_final = sum(sum(f.values()) * sum(p[j] for j in f) for p, f in zip(inner, shares))
    weights = []
    for p, f in zip(inner, shares):
        mass = sum(f.values())
        weights += [f[j] * in_final if j in f else mass * q for j, q in enumerate(p)]
    return weights


def _tree(read: Reader, texts: list[str]) -> list[float]:
    runs = groups(len(texts), min(len(LETTERS), -(-len(texts) // len(LETTERS))))
    outer = read(["One of: " + "; ".join(texts[i] for i in run) for run in runs])
    weights = []
    for run, share in zip(runs, outer):
        weights += [share * q for q in _combine(read, [texts[i] for i in run], "tree")]
    return weights
