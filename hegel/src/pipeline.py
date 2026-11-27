"""The compile pipeline: corpus -> tokens -> chain -> transformer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from src.chain import Chain, Counts, build_chain, count_contexts
from src.compiler import compile_chain
from src.config import HegelConfig
from src.corpus import tokenize_split
from src.model import HegelModel


@dataclass
class Compiled:
    """
    Everything one compile run produces.

    Attributes:
        text_length: Characters in the corpus file.
        tokens: The training part of the corpus as one token stream.
        held_out: The held-out part, never seen by tokenizer or chain; empty
            when the config holds nothing out.
        tokenizer: The fitted BPE tokenizer (encode / decode / vocab_size).
        counts: Raw next-token counts for every context.
        chain: The pruned backoff chain.
        model: The transformer compiled from the chain.
    """

    text_length: int
    tokens: List[int]
    held_out: List[int]
    tokenizer: Any
    counts: Counts
    chain: Chain
    model: HegelModel


def compile_from_config(config: HegelConfig) -> Compiled:
    """Run tokenize, chain and compile for a config. No gradient steps."""
    text = Path(config.corpus).read_text(encoding="utf-8")
    tokens, held_out, tokenizer = tokenize_split(
        text,
        config.vocab_size,
        config.holdout_fraction,
        config.holdout_block,
        config.holdout_seed,
    )
    counts = count_contexts(tokens, config.max_context)
    chain = build_chain(
        counts,
        int(tokenizer.vocab_size),
        config.max_context,
        config.floor,
        config.prune_threshold,
        config.smoothing,
    )
    return Compiled(
        text_length=len(text),
        tokens=tokens,
        held_out=held_out,
        tokenizer=tokenizer,
        counts=counts,
        chain=chain,
        model=compile_chain(
            chain,
            slot_dim=config.slot_dim,
            logit_dim=config.logit_dim,
            match_threshold=config.match_threshold,
            low_weight=config.low_weight,
            decoder_rounds=config.decoder_rounds,
            code_seed=config.code_seed,
        ),
    )
