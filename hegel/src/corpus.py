"""Corpus loading: one unbroken token stream from a text file."""

import importlib.util
import re
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

EOT_MARKER = "<|endoftext|>"

_BPE_PATH = (
    Path(__file__).resolve().parents[2] / "semantic" / "src" / "bpe_tokenizer.py"
)


def normalize(text: str) -> str:
    """
    Flatten a corpus into one stream of text.

    End-of-text markers are removed and every run of whitespace (newlines
    included) becomes a single space, so there are no paragraph boundaries.
    """
    return re.sub(r"\s+", " ", text.replace(EOT_MARKER, " ")).strip()


def _bpe_tokenizer_class() -> Any:
    """
    Load BPETokenizer from the sibling semantic experiment.

    Both experiments name their package `src`, so it is loaded by file path
    rather than imported.
    """
    spec = importlib.util.spec_from_file_location("semantic_bpe_tokenizer", _BPE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the BPE tokenizer from {_BPE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BPETokenizer


def split_stream(
    stream: str, holdout_fraction: float, block_chars: int = 1000, seed: int = 0
) -> Tuple[str, str]:
    """
    Hold out a random sample of blocks from a normalized stream.

    The stream is cut into blocks of about block_chars characters, at word
    boundaries. A random holdout_fraction of the blocks is held out; the
    rest, in their original order, are the training text. Both results are
    blocks joined end to end, so each has a few seams where text that was
    not adjacent now is.

    Returns:
        (training text, held-out text). Every block but the first keeps its
        leading space, so words encode as they would mid-stream. The
        held-out text is empty when holdout_fraction is 0 or the stream is
        a single block.
    """
    if holdout_fraction <= 0.0:
        return stream, ""
    cuts = [0]
    while True:
        cut = stream.find(" ", cuts[-1] + block_chars)
        if cut < 0:
            break
        cuts.append(cut)
    blocks = [stream[a:b] for a, b in zip(cuts, cuts[1:] + [len(stream)])]
    if len(blocks) < 2:
        return stream, ""
    held_count = min(len(blocks) - 1, max(1, round(holdout_fraction * len(blocks))))
    order = np.random.default_rng(seed).permutation(len(blocks))
    held = set(int(i) for i in order[:held_count])
    train = "".join(block for i, block in enumerate(blocks) if i not in held)
    held_out = "".join(block for i, block in enumerate(blocks) if i in held)
    return train, held_out


def new_tokenizer(vocab_size: int) -> Any:
    """An unfitted BPE tokenizer: fit it, or load a saved vocabulary into it."""
    return _bpe_tokenizer_class()(vocab_size=vocab_size)


def tokenize_split(
    text: str,
    vocab_size: int,
    holdout_fraction: float,
    holdout_block: int = 1000,
    holdout_seed: int = 0,
) -> Tuple[List[int], List[int], Any]:
    """
    Fit a BPE vocabulary on the training part of the text and encode both parts.

    The held-out part is never shown to the tokenizer's training.

    Returns:
        The training token stream, the held-out token stream, and the fitted
        tokenizer (encode / decode / vocab_size, where vocab_size is the size
        actually reached).
    """
    train, held_out = split_stream(
        normalize(text), holdout_fraction, holdout_block, holdout_seed
    )
    tokenizer = new_tokenizer(vocab_size)
    tokenizer.fit([train])
    return list(tokenizer.encode(train)), list(tokenizer.encode(held_out)), tokenizer


def tokenize(text: str, vocab_size: int) -> Tuple[List[int], Any]:
    """Fit a BPE vocabulary on the whole normalized text and encode it."""
    tokens, _, tokenizer = tokenize_split(text, vocab_size, 0.0)
    return tokens, tokenizer
