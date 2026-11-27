"""
The Markov chain: longest-match backoff over one token stream.

A context is the last few tokens up to and including the current one, oldest
first. Each kept context holds one next-token distribution. Prediction uses
the longest kept context that matches; the empty context (the unigram
distribution) always matches.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

Context = Tuple[int, ...]
Counts = Dict[Context, "Counter[int]"]

ROOT: Context = ()

SMOOTHINGS = ("none", "witten_bell")


def count_contexts(tokens: Sequence[int], max_context: int) -> Counts:
    """
    Count next tokens after every context of 0 to max_context tokens.

    The stream is unbroken: no start or end markers, and the final token has
    no successor so it contributes no count.
    """
    counts: Counts = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        nxt = tokens[i + 1]
        counts[ROOT][nxt] += 1
        for length in range(1, min(max_context, i + 1) + 1):
            counts[tuple(tokens[i - length + 1 : i + 1])][nxt] += 1
    return dict(counts)


def _log_distribution(
    counter: "Counter[int]", vocab_size: int, floor: float
) -> npt.NDArray[np.float64]:
    """Log of (1 - floor) * relative frequency + floor * uniform."""
    probs = np.full(vocab_size, floor / vocab_size, dtype=np.float64)
    total = sum(counter.values())
    for token, count in counter.items():
        probs[token] += (1.0 - floor) * count / total
    return np.log(probs)


@dataclass
class Chain:
    """
    A pruned backoff chain.

    Attributes:
        vocab_size: Number of token ids.
        max_context: Longest context length considered.
        floor: Uniform mass mixed into every distribution.
        smoothing: How a context's distribution was made; see build_chain.
        contexts: Kept contexts, shortest first; index 0 is the empty context.
        parents: For each kept context, the index of its longest kept proper
            suffix (what it backs off to); -1 for the empty context.
        occurrences: How often each kept context was followed by a token.
        log_probs: Next-token log-probabilities, one row per kept context.
        candidates: Number of distinct contexts seen, by length, before
            pruning.
    """

    vocab_size: int
    max_context: int
    floor: float
    smoothing: str
    contexts: List[Context]
    parents: List[int]
    occurrences: List[int]
    log_probs: npt.NDArray[np.float64]
    candidates: List[int]

    def __post_init__(self) -> None:
        self._index: Dict[Context, int] = {
            context: i for i, context in enumerate(self.contexts)
        }

    def match(self, history: Sequence[int]) -> int:
        """Index of the longest kept context that ends the history."""
        for length in range(min(self.max_context, len(history)), 0, -1):
            found = self._index.get(tuple(history[len(history) - length :]))
            if found is not None:
                return found
        return 0

    def match_stream(self, tokens: Sequence[int]) -> npt.NDArray[np.int64]:
        """For every position, the kept context predicting the next token."""
        matches = np.zeros(len(tokens), dtype=np.int64)
        for i in range(len(tokens)):
            matches[i] = self.match(tokens[max(0, i - self.max_context + 1) : i + 1])
        return matches

    def next_log_probs(self, history: Sequence[int]) -> npt.NDArray[np.float64]:
        """Next-token log-probabilities after the given history."""
        row: npt.NDArray[np.float64] = self.log_probs[self.match(history)]
        return row

    def match_counts(self) -> npt.NDArray[np.int64]:
        """
        How often each kept context was the longest match, so the one that
        predicted. A context's occurrences, less those of the kept contexts
        that back off directly to it.
        """
        counts = np.asarray(self.occurrences, dtype=np.int64).copy()
        for index in range(1, len(self.contexts)):
            counts[self.parents[index]] -= self.occurrences[index]
        return np.maximum(counts, 0)

    def kept_by_length(self) -> List[int]:
        """Number of kept contexts of each length, 0 to max_context."""
        kept = [0] * (self.max_context + 1)
        for context in self.contexts:
            kept[len(context)] += 1
        return kept


def build_chain(
    counts: Counts,
    vocab_size: int,
    max_context: int,
    floor: float = 1e-4,
    prune_threshold: float = 0.0,
    smoothing: str = "none",
) -> Chain:
    """
    Turn raw counts into a backoff chain, pruning contexts that add little.

    Contexts are visited shortest first. Each is compared with the longest
    proper suffix already kept, which is what would predict in its place.
    Its gain is occurrences * KL(context || that suffix), in nats: the total
    log-likelihood the corpus loses if the context is dropped. It is kept
    when the gain reaches prune_threshold. The empty context is always kept.

    smoothing decides what distribution a context holds:

    - "none": its own relative frequencies, with the uniform floor. A
      context seen once is certain of its one continuation.
    - "witten_bell": its relative frequencies mixed with the distribution
      of the context one token shorter (itself smoothed, down to the
      floored unigram). The context's own share is n / (n + t), for n
      occurrences and t distinct continuations, so a context seen once with
      one continuation gets half the say. The shorter context here is always
      the next one down, whether or not pruning keeps it.

    The longest kept match still predicts alone; smoothing only changes what
    it says.

    Raw counts are used as they are. The credit assignment of the original
    summed-count chain is not applied: under backoff only one context
    predicts at a time, so there is nothing to de-duplicate.
    """
    if smoothing not in SMOOTHINGS:
        raise ValueError(f"smoothing must be one of {SMOOTHINGS}, not {smoothing!r}")
    candidates = [0] * (max_context + 1)
    for context in counts:
        candidates[len(context)] += 1

    contexts: List[Context] = [ROOT]
    parents: List[int] = [-1]
    occurrences: List[int] = [sum(counts[ROOT].values())]
    rows: List[npt.NDArray[np.float64]] = [
        _log_distribution(counts[ROOT], vocab_size, floor)
    ]
    index: Dict[Context, int] = {ROOT: 0}
    # Smoothed probabilities of every context seen, kept or not, because a
    # context mixes with the one a token shorter even when that one is pruned.
    smoothed: Dict[Context, npt.NDArray[np.float64]] = {ROOT: np.exp(rows[0])}

    for context in sorted((c for c in counts if c), key=lambda c: (len(c), c)):
        parent = 0
        for length in range(len(context) - 1, 0, -1):
            found = index.get(context[len(context) - length :])
            if found is not None:
                parent = found
                break
        seen = sum(counts[context].values())
        if smoothing == "witten_bell":
            own = seen / (seen + len(counts[context]))
            probs = (1.0 - own) * smoothed[context[1:]]
            for token, count in counts[context].items():
                probs[token] += own * count / seen
            smoothed[context] = probs
            row = np.log(probs)
        else:
            row = _log_distribution(counts[context], vocab_size, floor)
        if prune_threshold > 0.0:
            gain = seen * float(np.sum(np.exp(row) * (row - rows[parent])))
            if gain < prune_threshold:
                continue
        index[context] = len(contexts)
        contexts.append(context)
        parents.append(parent)
        occurrences.append(seen)
        rows.append(row)

    return Chain(
        vocab_size=vocab_size,
        max_context=max_context,
        floor=floor,
        smoothing=smoothing,
        contexts=contexts,
        parents=parents,
        occurrences=occurrences,
        log_probs=np.stack(rows),
        candidates=candidates,
    )
