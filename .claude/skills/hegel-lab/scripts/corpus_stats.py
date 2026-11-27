"""
Size up a corpus before compiling it: tokens, distinct contexts by length,
and what the configured pruning keeps. Builds the chain but no model.

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python corpus_stats.py <config>
"""

import sys
import time
from typing import List

from lab import build_split, config_named, peak_gb


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    config = config_named(argv[1])
    started = time.perf_counter()
    built = build_split(config)
    print(f"Corpus: {config.corpus}, {built.characters} characters")
    print(f"  {len(built.tokens)} training tokens, {len(built.held_out)} held out")
    print(f"Contexts (prune threshold {config.prune_threshold}):")
    print("\n".join(built.by_length()))
    total = len(built.chain.contexts) - 1
    width = config.max_context * config.slot_dim + config.logit_dim
    if config.slot_dim == 0 or config.logit_dim == 0:
        width = built.chain.vocab_size * (config.max_context + 1)
    print(f"  kept in all: {total} (this is d_ff)")
    print(
        f"  model weights, roughly 2 * d_ff * d_model: {2 * total * width / 1e6:.0f} M"
    )
    print(f"  tokenized, counted and pruned in {time.perf_counter() - started:.0f} s")
    print(f"Peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
