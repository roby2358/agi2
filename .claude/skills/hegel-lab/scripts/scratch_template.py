"""
Starting point for a one-off measurement. Copy it to the scratchpad, edit,
and run it with:  just cc-py /path/to/copy.py [args]

cc-py runs from hegel/ with hegel/ and this scripts directory on the path,
so src.* and lab import without any sys.path edits, and corpus paths in the
configs resolve.
"""

import sys
from dataclasses import replace
from typing import List

from lab import config_named, peak_gb
from src.pipeline import compile_from_config
from src.verify import chain_score


def main(argv: List[str]) -> int:
    config = config_named(argv[1] if len(argv) > 1 else "corpus20_squeezed")
    compiled = compile_from_config(replace(config, holdout_fraction=0.1))
    nll, accuracy = chain_score(compiled.chain, compiled.held_out)
    print(f"contexts {len(compiled.chain.contexts) - 1}", flush=True)
    print(f"held-out chain NLL {nll:.3f}, top-1 {100 * accuracy:.1f}%")
    print(f"peak memory {peak_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
