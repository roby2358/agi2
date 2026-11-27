"""TOML configuration for a hegel compile run."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class HegelConfig:
    """
    Settings for one corpus -> chain -> transformer run.

    Args:
        corpus: Path to the corpus text file, relative to the working directory.
        vocab_size: BPE vocabulary size, including the 256 bytes and the
            tokenizer's special tokens.
        max_context: Longest context in tokens, counting the current token.
        floor: Probability mass each context spreads uniformly over the
            vocabulary, so unseen next tokens stay finite in log space.
        holdout_fraction: Share of the text, as randomly chosen blocks, that
            is kept away from the tokenizer and the chain and only used for
            scoring. 0 uses the whole corpus.
        holdout_block: Approximate block size in characters for the split.
        holdout_seed: Seed choosing which blocks are held out.
        prune_threshold: Minimum gain in nats (occurrences times KL from the
            backoff parent) for a context to be kept. 0 keeps every context.
        slot_dim: Width of each token vector in the residual stream; 0 means
            one-hot (the wide build).
        logit_dim: Width of the output logit code; 0 means the full
            vocabulary with exact log-probabilities.
        match_threshold: How close to a full match a context neuron needs
            before it fires at all, between 0 and 1. Used when slot_dim > 0.
        unseen_weight: Weight of unseen tokens in the output fit. Used when
            logit_dim > 0.
        decoder_rounds: Alternating least-squares rounds for the output
            factoring; 0 keeps a random decoder. Used when logit_dim > 0.
        code_seed: Seed for the random token vectors and decoder.
        seed: Seed for sampling.
        sample_count: Number of side-by-side samples to print.
        sample_tokens: Tokens generated per sample.
        checkpoint: Where to save the compiled weights; empty to skip.
    """

    corpus: str
    vocab_size: int = 512
    max_context: int = 3
    floor: float = 1e-4
    holdout_fraction: float = 0.0
    holdout_block: int = 1000
    holdout_seed: int = 0
    prune_threshold: float = 0.0
    slot_dim: int = 0
    logit_dim: int = 0
    match_threshold: float = 0.7
    unseen_weight: float = 0.01
    decoder_rounds: int = 4
    code_seed: int = 0
    seed: int = 0
    sample_count: int = 4
    sample_tokens: int = 80
    checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.max_context < 1:
            raise ValueError("max_context must be at least 1")
        if not 0.0 < self.floor < 1.0:
            raise ValueError("floor must be strictly between 0 and 1")
        if not 0.0 <= self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        if self.holdout_block < 1:
            raise ValueError("holdout_block must be at least 1")
        if self.prune_threshold < 0.0:
            raise ValueError("prune_threshold must not be negative")
        if self.slot_dim < 0 or self.logit_dim < 0:
            raise ValueError("slot_dim and logit_dim must not be negative")
        if not 0.0 <= self.match_threshold < 1.0:
            raise ValueError("match_threshold must be in [0, 1)")
        if not 0.0 < self.unseen_weight <= 1.0:
            raise ValueError("unseen_weight must be in (0, 1]")
        if self.decoder_rounds < 0:
            raise ValueError("decoder_rounds must not be negative")


def load_config(path: str | Path) -> HegelConfig:
    """Load a HegelConfig from a TOML file, rejecting unknown keys."""
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    known = {field.name for field in fields(HegelConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {', '.join(unknown)}")
    return HegelConfig(**data)
