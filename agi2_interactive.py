#!/usr/bin/env python3
"""
AGI2 Interactive Generation Script

Usage: uv run python agi2_interactive.py <config_file>
"""

import sys
from pathlib import Path

import torch

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.config_loader import get_config_value, load_config
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.generation import generate_interactive
from src.model import AGI2Model


def _load_model_and_tokenizer(model_path, device):
    """Load model and tokenizer from a checkpoint file."""
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary with 'model_state_dict'")

    config_obj = checkpoint.get("config")
    if config_obj is None:
        raise ValueError("No config found in checkpoint")

    model = AGI2Model(config_obj)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is None:
        raise ValueError("No tokenizer found in checkpoint")

    return model, tokenizer


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python agi2_interactive.py <config_file>")
        sys.exit(1)

    config_path = sys.argv[1]

    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    print("Checking CUDA availability for interactive generation...")
    check_cuda_availability(verbose=True)

    model_path = Path(get_config_value(config, "model_path"))
    max_length = get_config_value(config, "max_length", 100)
    temperature = get_config_value(config, "temperature", 0.8)
    device_choice = get_config_value(config, "device", "auto")

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    device = get_optimal_device(device_choice)
    print(f"Using device: {device}")

    model, tokenizer = _load_model_and_tokenizer(model_path, device)
    print(f"Model config: {model.config}")

    generate_interactive(
        model,
        tokenizer,
        max_length,
        temperature,
        device,
    )


if __name__ == "__main__":
    main()
