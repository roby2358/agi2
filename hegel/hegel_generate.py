"""
Give a compiled transformer a source string and print its continuation.

If the config names a `checkpoint` and hegel_build.py has saved weights
there, they are loaded. Otherwise the model is compiled fresh from the
config (no training). Either way it is built on the whole corpus: the
config's held-out split is for choosing settings, not for the model you use.

Usage (from hegel/):
    uv run python hegel_generate.py resources/corpus20.toml "source string"
    uv run python hegel_generate.py resources/corpus20.toml      # interactive

In interactive mode, type a source string and press Enter; an empty line or
Ctrl-D quits. Each response is a fresh random sample.
"""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from src.checkpoint import build_settings, load_weights
from src.config import HegelConfig, load_config
from src.generation import respond
from src.model import HegelModel
from src.pipeline import compile_from_config
from src.verify import model_predictor


def _model_for(config: HegelConfig) -> Tuple[HegelModel, Any, str]:
    """The model and tokenizer for a config, and a word on where they came from."""
    if config.checkpoint and Path(config.checkpoint).exists():
        model, tokenizer, settings = load_weights(config.checkpoint)
        if settings == build_settings(config):
            return model, tokenizer, f"Loaded {config.checkpoint}"
        print(
            f"{config.checkpoint} was built with other settings; compiling "
            "instead. Run hegel_build.py to refresh it.",
            file=sys.stderr,
        )
    elif config.checkpoint:
        print(
            f"No weights at {config.checkpoint}; compiling instead. "
            "Run hegel_build.py to save them.",
            file=sys.stderr,
        )
    compiled = compile_from_config(config)
    return compiled.model, compiled.tokenizer, f"Compiled from {config.corpus}"


def main(argv: List[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    config = replace(load_config(argv[1]), holdout_fraction=0.0)
    model, tokenizer, origin = _model_for(config)
    predict = model_predictor(model)
    rng = np.random.default_rng()

    def answer(source: str) -> str:
        return source + respond(
            tokenizer,
            predict,
            source,
            config.sample_tokens,
            rng,
            temperature=config.temperature,
        )

    if len(argv) == 3:
        print(answer(argv[2]))
        return 0

    print(f"{origin}: d_model {model.d_model}, d_ff {model.d_ff}. Empty line to quit.")
    while True:
        try:
            source = input("> ")
        except EOFError:
            print()
            break
        if not source.strip():
            break
        print(answer(source))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
