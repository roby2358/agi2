"""
Give a compiled transformer a source string and print its continuation.

The model is compiled fresh from the config each time (a few seconds, no
training), so there is no checkpoint to keep in sync.

Usage (from hegel/):
    uv run python hegel_generate.py resources/corpus20.toml "source string"
    uv run python hegel_generate.py resources/corpus20.toml      # interactive

In interactive mode, type a source string and press Enter; an empty line or
Ctrl-D quits. Each response is a fresh random sample.
"""

import sys
from typing import List

import numpy as np
from src.config import load_config
from src.generation import respond
from src.pipeline import compile_from_config


def main(argv: List[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    config = load_config(argv[1])
    compiled = compile_from_config(config)
    rng = np.random.default_rng()

    if len(argv) == 3:
        print(argv[2] + respond(compiled, argv[2], config.sample_tokens, rng))
        return 0

    model = compiled.model
    print(
        f"Compiled from {config.corpus}: d_model {model.d_model}, "
        f"d_ff {model.d_ff}. Empty line to quit."
    )
    while True:
        try:
            source = input("> ")
        except EOFError:
            print()
            break
        if not source.strip():
            break
        print(source + respond(compiled, source, config.sample_tokens, rng))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
