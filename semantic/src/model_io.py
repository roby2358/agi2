"""
Model Checkpoint Loading

Shared helper for loading a model and tokenizer from a training checkpoint,
used by the generation and interactive entry-point scripts for both the
transformer (AGI2Model) and RWKV (RWKVModel) pipelines.
"""

from pathlib import Path
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


def load_model_and_tokenizer(
    model_path: Union[str, Path],
    device: Union[str, torch.device],
    model_cls: Callable[..., nn.Module],
) -> Tuple[nn.Module, object]:
    """Load a model and tokenizer from a checkpoint file.

    Args:
        model_path: Path to the .pt checkpoint
        device: Device to map the checkpoint onto
        model_cls: Model class to construct from the checkpoint's config
            (e.g. AGI2Model or RWKVModel)

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If the checkpoint is malformed or was saved from a
            different model class than model_cls.
    """
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary with 'model_state_dict'")

    config_obj = checkpoint.get("config")
    if config_obj is None:
        raise ValueError("No config found in checkpoint")

    saved_type = checkpoint.get("model_type")
    expected_type = getattr(model_cls, "__name__", str(model_cls))
    if saved_type is not None and saved_type != expected_type:
        raise ValueError(
            f"Checkpoint was saved from {saved_type}, but this script loads "
            f"{expected_type}. Use the matching entry-point script."
        )

    model = model_cls(config_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is None:
        raise ValueError("No tokenizer found in checkpoint")

    return model, tokenizer
