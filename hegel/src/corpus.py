"""Corpus loading: one unbroken token stream from a text file."""

import importlib.util
import re
from pathlib import Path
from typing import Any, List, Tuple

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


def tokenize(text: str, vocab_size: int) -> Tuple[List[int], Any]:
    """
    Fit a BPE vocabulary on the normalized text and encode it.

    Returns:
        The token stream and the fitted tokenizer (encode / decode /
        vocab_size, where vocab_size is the size actually reached).
    """
    stream = normalize(text)
    tokenizer = _bpe_tokenizer_class()(vocab_size=vocab_size)
    tokenizer.fit([stream])
    return list(tokenizer.encode(stream)), tokenizer
