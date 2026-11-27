"""Helpers shared by the hegel-lab scripts. Run them from hegel/ (see justfile)."""

import dataclasses
import resource
from pathlib import Path
from typing import Any, Dict, List

from src.chain import Chain, build_chain, count_contexts
from src.config import HegelConfig, load_config
from src.corpus import tokenize_split


@dataclasses.dataclass(frozen=True)
class SplitChain:
    """
    A config's first split and the chain built on its training part.

    Attributes:
        characters: Characters in the corpus file.
        tokens: The training part as one token stream.
        held_out: The held-out part; empty when the config holds nothing out.
        chain: The chain as configured (pruned).
        repeated: Distinct contexts seen twice or more, by length.
    """

    characters: int
    tokens: List[int]
    held_out: List[int]
    chain: Chain
    repeated: Dict[int, int]

    def by_length(self) -> List[str]:
        """One line per context length: distinct, seen twice or more, kept."""
        kept = self.chain.kept_by_length()
        return [
            f"  length {length}: {self.chain.candidates[length]} distinct, "
            f"{self.repeated.get(length, 0)} seen twice or more, {kept[length]} kept"
            for length in range(1, len(kept))
        ]


def build_split(config: HegelConfig) -> SplitChain:
    """Tokenize the config's first split and build its chain. No model."""
    text = Path(config.corpus).read_text(encoding="utf-8")
    tokens, held_out, tokenizer = tokenize_split(
        text,
        config.vocab_size,
        config.holdout_fraction,
        config.holdout_block,
        config.holdout_seed,
    )
    counts = count_contexts(tokens, config.max_context)
    repeated: Dict[int, int] = {}
    for context, nexts in counts.items():
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
    return SplitChain(len(text), tokens, held_out, chain, repeated)


def config_named(name: str) -> HegelConfig:
    """Load resources/<name>.toml; a path to a TOML file works too."""
    path = name if name.endswith(".toml") else f"resources/{name}.toml"
    return load_config(path)


def with_setting(config: HegelConfig, key: str, raw: str) -> HegelConfig:
    """A copy of config with one field set from a command-line string."""
    kinds = {f.name: type(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if key not in kinds:
        raise SystemExit(f"No setting '{key}'. Settings: {', '.join(sorted(kinds))}")
    value: Any = kinds[key](raw)
    return dataclasses.replace(config, **{key: value})


def peak_gb() -> float:
    """Peak memory of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def table(header: List[str], rows: List[List[str]]) -> str:
    """Rows as left-aligned columns."""
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in [header] + rows]
    return "\n".join(line.rstrip() for line in lines)
