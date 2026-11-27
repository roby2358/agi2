"""Save and load compiled weights, so a model need not be recompiled to use."""

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from src.config import HegelConfig
from src.corpus import new_tokenizer
from src.model import HegelModel

FORMAT = 1

# Settings that do not change the weights of a whole-corpus build.
_NOT_IN_WEIGHTS = (
    "holdout_fraction",
    "holdout_block",
    "holdout_seed",
    "holdout_seeds",
    "seed",
    "sample_count",
    "sample_tokens",
    "temperature",
    "checkpoint",
)


def build_settings(config: HegelConfig) -> Dict[str, Any]:
    """The config values that decide the weights of a whole-corpus build."""
    settings = asdict(config)
    for name in _NOT_IN_WEIGHTS:
        del settings[name]
    return settings


def save_weights(
    path: str | Path, model: HegelModel, tokenizer: Any, config: HegelConfig
) -> None:
    """
    Write the model, its tokenizer and the settings it was built with.

    Args:
        path: File to write; parent directories are created.
        model: The compiled transformer.
        tokenizer: The fitted BPE tokenizer (must offer save_vocab).
        config: The config the model was compiled from.
    """
    with tempfile.TemporaryDirectory() as folder:
        vocab_file = Path(folder) / "tokenizer.json"
        tokenizer.save_vocab(str(vocab_file))
        tokenizer_json = vocab_file.read_text(encoding="utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "shape": {
                "vocab_size": model.vocab_size,
                "d_model": model.d_model,
                "n_heads": model.n_heads,
                "d_head": model.d_head,
                "d_ff": model.d_ff,
                "max_offset": model.max_offset,
            },
            "state": model.state_dict(),
            "tokenizer": tokenizer_json,
            "settings": build_settings(config),
        },
        path,
    )


def load_weights(path: str | Path) -> Tuple[HegelModel, Any, Dict[str, Any]]:
    """
    Read what save_weights wrote.

    Returns:
        The model, the tokenizer, and the settings the model was built with
        (compare with build_settings to see whether a config still matches).

    Raises:
        ValueError: If the file is not in the current format.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != FORMAT:
        raise ValueError(f"{path} is not a hegel weights file of format {FORMAT}")
    model = HegelModel(**saved["shape"])
    model.load_state_dict(saved["state"])
    tokenizer = new_tokenizer(model.vocab_size)
    with tempfile.TemporaryDirectory() as folder:
        vocab_file = Path(folder) / "tokenizer.json"
        vocab_file.write_text(saved["tokenizer"], encoding="utf-8")
        tokenizer.load_vocab(str(vocab_file))
    return model, tokenizer, dict(saved["settings"])
