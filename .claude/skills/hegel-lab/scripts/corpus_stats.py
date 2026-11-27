"""
Size up a corpus before compiling it: tokens, distinct contexts by length,
and what the configured pruning keeps. Builds the chain but no model.

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python corpus_stats.py <config>
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

from lab import config_named, peak_gb
from src.chain import build_chain, count_contexts
from src.corpus import tokenize_split


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    config = config_named(argv[1])
    started = time.perf_counter()
    text = Path(config.corpus).read_text(encoding="utf-8")
    tokens, held_out, tokenizer = tokenize_split(
        text,
        config.vocab_size,
        config.holdout_fraction,
        config.holdout_block,
        config.holdout_seed,
    )
    print(f"Corpus: {config.corpus}, {len(text)} characters")
    print(f"  {len(tokens)} training tokens, {len(held_out)} held out")
    print(f"  tokenized in {time.perf_counter() - started:.0f} s", flush=True)

    started = time.perf_counter()
    counts = count_contexts(tokens, config.max_context)
    distinct: Dict[int, int] = {}
    repeated: Dict[int, int] = {}
    for context, nexts in counts.items():
        if not context:
            continue
        distinct[len(context)] = distinct.get(len(context), 0) + 1
        if sum(nexts.values()) >= 2:
            repeated[len(context)] = repeated.get(len(context), 0) + 1
    chain = build_chain(
        counts,
        int(tokenizer.vocab_size),
        config.max_context,
        config.floor,
        config.prune_threshold,
        config.smoothing,
        config.discount,
    )
    kept = chain.kept_by_length()
    print(f"Contexts (prune threshold {config.prune_threshold}):")
    for length in sorted(distinct):
        print(
            f"  length {length}: {distinct[length]} distinct, "
            f"{repeated.get(length, 0)} seen twice or more, {kept[length]} kept"
        )
    total = len(chain.contexts) - 1
    width = config.max_context * config.slot_dim + config.logit_dim
    if config.slot_dim == 0 or config.logit_dim == 0:
        width = int(tokenizer.vocab_size) * (config.max_context + 1)
    print(f"  kept in all: {total} (this is d_ff)")
    print(
        f"  model weights, roughly 2 * d_ff * d_model: {2 * total * width / 1e6:.0f} M"
    )
    print(f"  counted and pruned in {time.perf_counter() - started:.0f} s")
    print(f"Peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
