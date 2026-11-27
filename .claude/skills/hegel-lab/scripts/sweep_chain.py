"""
Try several values of one chain setting and score each on held-out text.
Chain only, no model, so it is quick. Each value is scored on the config's
holdout_seeds splits (mean, with the range in brackets).

Usage (from hegel/, with hegel/ on PYTHONPATH):
    python sweep_chain.py <config> <setting> <value> [value...]

Settings that matter here: prune_threshold, max_context, vocab_size,
smoothing, floor, holdout_fraction, holdout_seeds.
"""

import sys
import time
from typing import List

from lab import config_named, table, with_setting
from src.holdout import score_seeds


def _spread(values: List[float], digits: int = 3) -> str:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def main(argv: List[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    base, key = config_named(argv[1]), argv[2]
    if base.holdout_fraction == 0.0 and key != "holdout_fraction":
        raise SystemExit("This config holds nothing out; set holdout_fraction.")
    print(f"{argv[1]}: {key} over {base.holdout_seeds} split(s), held-out text")
    rows: List[List[str]] = []
    for raw in argv[3:]:
        started = time.perf_counter()
        scores = score_seeds(with_setting(base, key, raw))
        rows.append(
            [
                raw,
                f"{sum(s.contexts for s in scores) / len(scores):.0f}",
                _spread([s.chain_nll for s in scores]),
                f"{100 * sum(s.chain_accuracy for s in scores) / len(scores):.1f}%",
                _spread([s.unpruned_nll for s in scores]),
                f"{time.perf_counter() - started:.0f}",
            ]
        )
        print(f"  done {key} = {raw}", file=sys.stderr, flush=True)
    header = [key, "contexts", "NLL (nats)", "top-1", "unpruned NLL", "seconds"]
    print(table(header, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
