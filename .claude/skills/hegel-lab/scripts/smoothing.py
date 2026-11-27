"""
Held-out score of unpruned chains at context lengths 1..N under several
smoothings, built on the lean tables of long_context.py.

Usage: just cc-smoothing <config> [max_context]

  wb k   Witten-Bell with the backoff share scaled: own = n / (n + k t)
  ad D   absolute discounting, interpolated, raw counts at every level
  kn D   Kneser-Ney: the same, lower levels count distinct left neighbours
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from lab import config_named, peak_gb
from long_context import V_BITS, table
from src.corpus import tokenize_split

Table = Tuple["np.ndarray", ...]


def summarize(pair: "np.ndarray", count: "np.ndarray") -> Table:
    ctx = pair >> V_BITS
    ctx_keys, start, distinct = np.unique(ctx, return_index=True, return_counts=True)
    return pair, count, ctx_keys, start, distinct, np.add.reduceat(count, start)


def continuation(raw_above: Table, length: int) -> Table:
    """Level-`length` table counting distinct left neighbours of (context, next)."""
    mask = (1 << (V_BITS * (length + 1))) - 1
    pair, count = np.unique(raw_above[0] & mask, return_counts=True)
    return summarize(pair, count)


def apply(
    probs: "np.ndarray",
    t: Table,
    key: "np.ndarray",
    usable: "np.ndarray",
    method: str,
    param: float,
    vocab: int,
) -> "np.ndarray":
    """probs one level up: rows whose context is in the table get mixed."""
    pair, count, ctx_keys, start, distinct, total = t
    at = np.searchsorted(ctx_keys, key)
    at[at == len(ctx_keys)] = 0
    idx = np.nonzero(usable & (ctx_keys[at] == key))[0]
    c = at[idx]
    d, n = distinct[c].astype(np.float64), total[c].astype(np.float64)
    reps = distinct[c]
    flat = np.repeat(start[c], reps) + (
        np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    )
    row_of = np.repeat(np.arange(len(idx)), reps)
    counts = count[flat].astype(np.float64)
    if method == "wb":
        denom = n + param * d
        mult, add = param * d / denom, counts / np.repeat(denom, reps)
    else:
        mult = param * d / n
        add = np.maximum(counts - param, 0.0) / np.repeat(n, reps)
    out = probs.copy()
    new = probs[idx] * mult[:, None]
    new[row_of, pair[flat] & (vocab - 1)] += add
    out[idx] = new
    return out


def main(argv: List[str]) -> int:
    config = config_named(argv[1])
    top = int(argv[2]) if len(argv) > 2 else 5
    vocab = config.vocab_size
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

    def floored(counts: "np.ndarray") -> "np.ndarray":
        mixed: "np.ndarray" = (1.0 - config.floor) * counts / counts.sum()
        return mixed + config.floor / vocab

    raw = {length: table(train, length) for length in range(1, top + 1)}
    cont = {length: continuation(raw[length + 1], length) for length in range(1, top)}
    uni_raw = floored(np.bincount(train[1:], minlength=vocab).astype(np.float64))
    left = np.unique(raw[1][0] & (vocab - 1), return_counts=True)
    uni_cont = np.zeros(vocab)
    uni_cont[left[0]] = left[1]
    uni_cont = floored(uni_cont)
    print("tables built", flush=True)

    variants = [("wb", k) for k in (1.0, 2.0, 3.0, 5.0, 8.0)]
    variants += [("ad", d) for d in (0.5, 0.7, 0.8, 0.9)]
    variants += [("kn", d) for d in (0.6, 0.7, 0.8, 0.9)]
    targets = held[1:]
    n = len(targets)
    nll: Dict[Tuple[str, float], "np.ndarray"] = {
        v: np.zeros(top + 1) for v in variants
    }
    hit: Dict[Tuple[str, float], "np.ndarray"] = {
        v: np.zeros(top + 1) for v in variants
    }
    chunk = 20000
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        rows = np.arange(lo, hi)
        here = np.arange(hi - lo)
        keys, usable = {}, {}
        for length in range(1, top + 1):
            key = np.zeros(hi - lo, dtype=np.int64)
            for j in range(length):
                key = (key << V_BITS) | held[np.maximum(rows - (length - 1) + j, 0)]
            keys[length], usable[length] = key, rows >= length - 1
        for variant in variants:
            method, param = variant
            kn = method == "kn"
            carry = np.tile(uni_cont if kn else uni_raw, (hi - lo, 1))
            for length in range(1, top + 1):
                args = (keys[length], usable[length], method, param, vocab)
                final = apply(carry, raw[length], *args)
                nll[variant][length] -= np.log(final[here, targets[lo:hi]]).sum()
                hit[variant][length] += (final.argmax(axis=1) == targets[lo:hi]).sum()
                if not kn:
                    carry = final
                elif length < top:
                    carry = apply(carry, cont[length], *args)
    lengths = list(range(1, top + 1))
    print("held-out NLL (top-1 %) by max_context")
    print("smoothing   " + "".join(f"{length:<15}" for length in lengths))
    for variant in variants:
        cells = "".join(
            f"{nll[variant][length] / n:.3f} ({100 * hit[variant][length] / n:.1f})  "
            for length in lengths
        )
        print(f"{variant[0]} {variant[1]:<8} {cells}")
    print(f"peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
