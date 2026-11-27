"""
Held-out score of the unpruned Witten-Bell chain at context lengths 1..N,
without building dense rows for every context (src.chain needs 4 KB per
context seen, which does not fit at context 4 or 5 on Hegel).

Usage: just cc-context <config> [max_context]

Check: the max_context 3 row must match the unpruned chain in hegel/CLAUDE.md.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from lab import config_named, peak_gb
from src.corpus import tokenize_split

V_BITS = 9  # vocabulary 512


def keys_for(tokens: "np.ndarray", length: int) -> "np.ndarray":
    """Key of the context of `length` tokens ending at each position >= length-1."""
    key = np.zeros(len(tokens) - length + 1, dtype=np.int64)
    for j in range(length):
        key = (key << V_BITS) | tokens[j : len(tokens) - length + 1 + j]
    return key


def table(tokens: "np.ndarray", length: int) -> Tuple["np.ndarray", ...]:
    """Sorted unique (context, next) pairs with counts, and per-context totals."""
    key = keys_for(tokens[:-1], length)
    nxt = tokens[length:]
    pair, count = np.unique((key << V_BITS) | nxt, return_counts=True)
    ctx = pair >> V_BITS
    ctx_keys, start, distinct = np.unique(ctx, return_index=True, return_counts=True)
    total = np.add.reduceat(count, start)
    return pair, count, ctx_keys, start, distinct, total


def main(argv: List[str]) -> int:
    config = config_named(argv[1])
    top = int(argv[2]) if len(argv) > 2 else 5
    vocab = config.vocab_size
    assert vocab == 1 << V_BITS
    text = Path(config.corpus).read_text(encoding="utf-8")
    train_list, held_list, _ = tokenize_split(
        text,
        vocab,
        config.holdout_fraction,
        config.holdout_block,
        config.holdout_seed,
    )
    train = np.asarray(train_list, dtype=np.int64)
    held = np.asarray(held_list, dtype=np.int64)
    print(f"{len(train)} training tokens, {len(held)} held out", flush=True)

    uni = np.bincount(train[1:], minlength=vocab).astype(np.float64)
    uni = (1.0 - config.floor) * uni / uni.sum() + config.floor / vocab

    tables = []
    for length in range(1, top + 1):
        started = time.perf_counter()
        t = table(train, length)
        tables.append(t)
        print(
            f"length {length}: {len(t[2])} distinct contexts, "
            f"{int((t[5] >= 2).sum())} seen twice or more, "
            f"{int((t[5] >= 3).sum())} three times or more "
            f"({time.perf_counter() - started:.0f} s)",
            flush=True,
        )

    targets = held[1:]
    n = len(targets)
    nll = np.zeros(top + 1)
    hits = np.zeros(top + 1)
    matched = np.zeros(top + 1)
    chunk = 20000
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        rows = np.arange(lo, hi)
        probs = np.tile(uni, (hi - lo, 1))
        for length in range(1, top + 1):
            pair, count, ctx_keys, start, distinct, total = tables[length - 1]
            usable = rows >= length - 1  # position i has i + 1 tokens of history
            key = np.zeros(hi - lo, dtype=np.int64)
            for j in range(length):
                key = (key << V_BITS) | held[np.maximum(rows - (length - 1) + j, 0)]
            at = np.searchsorted(ctx_keys, key)
            at[at == len(ctx_keys)] = 0
            found = usable & (ctx_keys[at] == key)
            idx = np.nonzero(found)[0]
            c = at[idx]
            t, tot = distinct[c].astype(np.float64), total[c].astype(np.float64)
            new = probs[idx] * (t / (tot + t))[:, None]
            reps = distinct[c]
            flat = np.repeat(start[c], reps) + (
                np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
            )
            row_of = np.repeat(np.arange(len(idx)), reps)
            new[row_of, pair[flat] & (vocab - 1)] += count[flat] / np.repeat(
                tot + t, reps
            )
            probs[idx] = new
            nll[length] -= np.log(probs[np.arange(hi - lo), targets[lo:hi]]).sum()
            hits[length] += (probs.argmax(axis=1) == targets[lo:hi]).sum()
            matched[length] += found.sum()
    print("max_context  held-out NLL  top-1   positions matching at full length")
    for length in range(1, top + 1):
        print(
            f"{length:<11}  {nll[length] / n:.3f}         "
            f"{100 * hits[length] / n:.1f}%   {100 * matched[length] / n:.1f}%"
        )
    print(f"peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
