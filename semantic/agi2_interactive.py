#!/usr/bin/env python3
"""
AGI2 Interactive Generation Script

Usage: uv run python agi2_interactive.py <config_file>
"""

import sys
from pathlib import Path

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.config_loader import get_config_value, get_sources_list, load_config
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.generation import build_corpus_token_mask, generate_interactive
from src.model import AGI2Model
from src.model_io import load_model_and_tokenizer


def main(model_cls=AGI2Model):
    if len(sys.argv) != 2:
        script = Path(sys.argv[0]).name
        print(f"Usage: uv run python {script} <config_file>")
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

    model, tokenizer = load_model_and_tokenizer(model_path, device, model_cls)
    print(f"Model config: {model.config}")

    vocab_size = model.token_embeddings.embedding.num_embeddings
    allowed_mask = build_corpus_token_mask(
        get_sources_list(config), tokenizer, vocab_size, device
    )
    if allowed_mask is not None:
        print(
            f"Restricting generation to {int(allowed_mask.sum())} "
            f"corpus tokens (of {vocab_size})"
        )

    generate_interactive(
        model,
        tokenizer,
        max_length,
        temperature,
        device,
        allowed_mask,
    )


if __name__ == "__main__":
    main()
