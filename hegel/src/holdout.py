"""Score the chain on several random held-out splits."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List

from src.chain import build_chain, count_contexts
from src.config import HegelConfig
from src.corpus import tokenize_split
from src.verify import chain_score, unigram_nll


@dataclass(frozen=True)
class HoldoutScore:
    """
    Held-out results for one split. NLL is in nats per token; accuracy is
    how often the most likely token is the actual next one.

    Attributes:
        seed: The holdout_seed of this split.
        tokens: Number of held-out tokens.
        unigram_nll: The unigram distribution alone.
        short_nll, short_accuracy: A chain with one token of context.
        unpruned_nll, unpruned_accuracy: The full-context chain, unpruned.
        chain_nll, chain_accuracy: The chain as configured.
        contexts: Contexts the configured chain keeps.
    """

    seed: int
    tokens: int
    unigram_nll: float
    short_nll: float
    short_accuracy: float
    unpruned_nll: float
    unpruned_accuracy: float
    chain_nll: float
    chain_accuracy: float
    contexts: int


def score_split(config: HegelConfig, text: str) -> HoldoutScore:
    """Build the chain on one split and score its held-out part."""
    tokens, held_out, tokenizer = tokenize_split(
        text,
        config.vocab_size,
        config.holdout_fraction,
        config.holdout_block,
        config.holdout_seed,
    )
    vocab = int(tokenizer.vocab_size)
    counts = count_contexts(tokens, config.max_context)

    def build(max_context: int, threshold: float) -> tuple[float, float, int]:
        used = (
            counts
            if max_context == config.max_context
            else {c: n for c, n in counts.items() if len(c) <= max_context}
        )
        chain = build_chain(
            used, vocab, max_context, config.floor, threshold, config.smoothing
        )
        nll, accuracy = chain_score(chain, held_out)
        return nll, accuracy, len(chain.contexts) - 1

    short = build(1, 0.0)
    unpruned = build(config.max_context, 0.0)
    configured = build(config.max_context, config.prune_threshold)
    root = build_chain(
        {(): counts[()]}, vocab, config.max_context, config.floor, 0.0, "none"
    )
    return HoldoutScore(
        seed=config.holdout_seed,
        tokens=len(held_out),
        unigram_nll=unigram_nll(root, held_out),
        short_nll=short[0],
        short_accuracy=short[1],
        unpruned_nll=unpruned[0],
        unpruned_accuracy=unpruned[1],
        chain_nll=configured[0],
        chain_accuracy=configured[1],
        contexts=configured[2],
    )


def score_seeds(config: HegelConfig) -> List[HoldoutScore]:
    """Score config.holdout_seeds splits, starting at config.holdout_seed."""
    text = Path(config.corpus).read_text(encoding="utf-8")
    return [
        score_split(replace(config, holdout_seed=config.holdout_seed + offset), text)
        for offset in range(config.holdout_seeds)
    ]
