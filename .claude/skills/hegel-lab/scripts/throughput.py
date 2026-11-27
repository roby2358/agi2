"""
Time saved weights: how long they take to load, and how long a forward pass
over a stretch of tokens takes.

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python throughput.py <config> [tokens]      # tokens defaults to 1024
"""

import sys
import time
from pathlib import Path
from typing import List

import torch
from lab import config_named, peak_gb
from src.checkpoint import load_weights


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    config = config_named(argv[1])
    length = int(argv[2]) if len(argv) > 2 else 1024
    if not config.checkpoint or not Path(config.checkpoint).exists():
        raise SystemExit(f"No saved weights for {argv[1]}; run: just build {argv[1]}")
    started = time.perf_counter()
    model, tokenizer, _ = load_weights(config.checkpoint)
    size = Path(config.checkpoint).stat().st_size / 1e6
    print(
        f"{config.checkpoint}: {size:.0f} MB, loaded in "
        f"{time.perf_counter() - started:.1f} s"
    )
    print(f"  d_model {model.d_model}, d_ff {model.d_ff}")
    tokens = torch.randint(0, int(tokenizer.vocab_size), (1, length))
    with torch.no_grad():
        for label in ("first pass", "second pass"):
            started = time.perf_counter()
            model(tokens)
            seconds = time.perf_counter() - started
            print(
                f"  {label}: {length} tokens in {seconds:.2f} s "
                f"({1000 * seconds / length:.2f} ms per token)"
            )
    print(f"Peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
