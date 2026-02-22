#!/usr/bin/env python3
"""
AGI2 Beam Search Generation Script

Usage: uv run python agi2_generate_beam.py <config_file> [prompt]
"""

import sys
from pathlib import Path

import torch

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.config_loader import get_config_value, load_config
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.generation import generate_with_beam_search
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
    if len(sys.argv) < 2:
        print("Usage: uv run python agi2_generate_beam.py <config_file> [prompt]")
        sys.exit(1)

    config_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The future of AI is"

    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    print("Checking CUDA availability for beam search generation...")
    check_cuda_availability(verbose=True)

    model_path = Path(get_config_value(config, "model_path"))
    max_length = get_config_value(config, "max_length", 50)
    beam_size = get_config_value(config, "beam_size", 5)
    temperature = get_config_value(config, "temperature", 0.8)
    device_choice = get_config_value(config, "device", "auto")
    model_seed = get_config_value(config, "model_seed", "")

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    device = get_optimal_device(device_choice)
    print(f"Using device: {device}")

    model, tokenizer = _load_model_and_tokenizer(model_path, device)
    print(f"Model config: {model.config}")

    full_prompt = model_seed + prompt if model_seed else prompt
    print(f"Prompt: {len(full_prompt)} characters")
    print(f"Max length: {max_length}, Beam size: {beam_size}")
    print("-" * 50)

    results = generate_with_beam_search(
        model,
        full_prompt,
        max_length,
        beam_size,
        temperature,
        tokenizer,
        device,
    )

    print("Generated text (beam search):")
    for i, text in enumerate(results):
        print(f"\nBeam {i + 1}:")
        print(text)


if __name__ == "__main__":
    main()
