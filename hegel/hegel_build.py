"""
Compile a transformer on the whole corpus and save its weights.

The weights go to the config's `checkpoint` path, together with the
tokenizer. hegel_generate.py then loads them instead of recompiling. Nothing
is held out here: the held-out split is for hegel_compile.py, which measures.

Usage (from hegel/):
    uv run python hegel_build.py resources/hegel.toml
"""

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List

from src.checkpoint import save_weights
from src.config import load_config
from src.pipeline import compile_from_config


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    config = replace(load_config(argv[1]), holdout_fraction=0.0)
    if not config.checkpoint:
        print(f"{argv[1]} has no `checkpoint` path to save the weights to.")
        return 2
    started = time.perf_counter()

    compiled = compile_from_config(config)
    model = compiled.model
    print(f"Corpus: {config.corpus}")
    print(
        f"  {compiled.text_length} characters, {len(compiled.tokens)} tokens, "
        f"vocabulary {compiled.chain.vocab_size}"
    )
    print(f"Chain: {len(compiled.chain.contexts) - 1} contexts kept")
    print(f"Model: d_model {model.d_model}, heads {model.n_heads}, d_ff {model.d_ff}")
    print(
        f"  weights {model.parameter_count()}, "
        f"of which nonzero {model.nonzero_count()}"
    )
    save_weights(config.checkpoint, model, compiled.tokenizer, config)
    size = Path(config.checkpoint).stat().st_size / 1e6
    print(f"Saved to {config.checkpoint} ({size:.0f} MB)")
    print(f"Done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
