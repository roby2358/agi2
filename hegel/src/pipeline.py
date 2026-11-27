"""The compile pipeline: corpus -> tokens -> chain -> transformer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from src.chain import Chain, Counts, build_chain, count_contexts
from src.compiler import compile_chain
from src.config import HegelConfig
from src.corpus import tokenize
from src.model import HegelModel


@dataclass
class Compiled:
    """
    Everything one compile run produces.

    Attributes:
        text_length: Characters in the corpus file.
        tokens: The corpus as one token stream.
        tokenizer: The fitted BPE tokenizer (encode / decode / vocab_size).
        counts: Raw next-token counts for every context.
        chain: The pruned backoff chain.
        model: The transformer compiled from the chain.
    """

    text_length: int
    tokens: List[int]
    tokenizer: Any
    counts: Counts
    chain: Chain
    model: HegelModel


def compile_from_config(config: HegelConfig) -> Compiled:
    """Run tokenize, chain and compile for a config. No gradient steps."""
    text = Path(config.corpus).read_text(encoding="utf-8")
    tokens, tokenizer = tokenize(text, config.vocab_size)
    counts = count_contexts(tokens, config.max_context)
    chain = build_chain(
        counts,
        int(tokenizer.vocab_size),
        config.max_context,
        config.floor,
        config.prune_threshold,
    )
    return Compiled(
        text_length=len(text),
        tokens=tokens,
        tokenizer=tokenizer,
        counts=counts,
        chain=chain,
        model=compile_chain(
            chain,
            slot_dim=config.slot_dim,
            logit_dim=config.logit_dim,
            match_threshold=config.match_threshold,
            unseen_weight=config.unseen_weight,
            decoder_rounds=config.decoder_rounds,
            code_seed=config.code_seed,
        ),
    )
