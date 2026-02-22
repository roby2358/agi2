#!/usr/bin/env python3
"""
AGI2 Training Script

This script trains an AGI2 model on text data.
Usage: python agi2_train.py <config_file>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from src.basic_tokenizer import BasicTokenizer
from src.config import AGI2Config
from src.config_loader import get_config_value, get_sources_list, get_training_config
from src.cuda_utils import check_cuda_availability, get_optimal_device
from src.model import AGI2Model
from src.tiktoken_tokenizer import TiktokenTokenizer
from src.training import train_model


def main():
    if len(sys.argv) < 2:
        print("Usage: python agi2_train.py <config_file>")
        print("Example: python agi2_train.py resources/small_model.toml")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        # Load configuration from TOML file
        config = get_training_config(config_path)
        print(f"Loaded configuration from: {config_path}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Check CUDA availability
    print("Checking CUDA availability for training...")
    cuda_status = check_cuda_availability(verbose=True)

    # Extract configuration values
    sources = get_sources_list(config)
    model_name = get_config_value(config, "model_name")
    epochs = get_config_value(config, "epochs", 10)
    batch_size = get_config_value(config, "batch_size", 4)
    learning_rate = get_config_value(config, "learning_rate", 1e-4)
    seq_len_start = get_config_value(config, "seq_len_start", 2)
    seq_len_end = get_config_value(
        config, "seq_len_end", get_config_value(config, "model_positions", 512)
    )
    device_choice = get_config_value(config, "device", "auto")
    resume_path = get_config_value(config, "resume", "")

    # Cosine similarity training parameters
    geometric_ratio = get_config_value(config, "geometric_ratio", 0.7)
    anchor_ratio = get_config_value(config, "anchor_ratio", 0.3)
    sigmoid_scale_start = get_config_value(config, "sigmoid_scale_start", 3.0)
    sigmoid_scale_end = get_config_value(config, "sigmoid_scale_end", 10.0)
    early_stop_patience = get_config_value(config, "early_stop_patience", 20)

    # Validate source paths
    for source_path in sources:
        if not Path(source_path).exists():
            print(f"Error: Source file not found: {source_path}")
            sys.exit(1)

    print(f"Training with {len(sources)} source(s):")
    for source_path in sources:
        print(f"  - {source_path}")

    # Determine device
    device = get_optimal_device(device_choice)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs("trained", exist_ok=True)

    # Build tokenizer
    tokenizer_type = get_config_value(config, "tokenizer", "tiktoken")
    print(f"Tokenizer: {tokenizer_type}")

    if tokenizer_type == "tiktoken":
        tokenizer = TiktokenTokenizer()
        actual_vocab_size = tokenizer.vocab_size
        print(f"Loaded tiktoken GPT-2 vocabulary: {actual_vocab_size} tokens")
    elif tokenizer_type == "char":
        print("Building character vocabulary from corpus...")
        with open(sources[0], "r", encoding="utf-8") as f:
            sample_text = f.read()[:50000]
        tokenizer = BasicTokenizer()
        tokenizer.fit([sample_text])
        actual_vocab_size = tokenizer.vocab_size
        print(f"Character vocabulary built: {actual_vocab_size} tokens")
    else:
        print(f"Error: Unknown tokenizer type: {tokenizer_type}")
        sys.exit(1)

    # Create model configuration
    model_config = AGI2Config(
        vocab_size=actual_vocab_size,
        n_positions=seq_len_end,
        n_ctx=seq_len_end,
        n_embd=get_config_value(config, "model_embd", 768),
        n_layer=get_config_value(config, "model_layer", 12),
        n_head=get_config_value(config, "model_head", 12),
        activation_function=get_config_value(config, "model_activation", "gelu"),
        resid_pdrop=get_config_value(config, "model_dropout", 0.1),
        embd_pdrop=get_config_value(config, "model_dropout", 0.1),
        attn_pdrop=get_config_value(config, "model_dropout", 0.1),
    )

    # Create model
    model = AGI2Model(model_config)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model state from epoch {checkpoint['epoch']}")

        # Load tokenizer if it exists in checkpoint
        if "tokenizer" in checkpoint:
            tokenizer = checkpoint["tokenizer"]
            print("Loaded tokenizer from checkpoint")

        # Calculate starting epoch
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Compile model if requested (requires PyTorch 2.0+)
    use_compile = get_config_value(config, "use_compile", False)
    if use_compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("Model compiled successfully")

    # Train the model
    try:
        training_history = train_model(
            model=model,
            tokenizer=tokenizer,
            sources=sources,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seq_len_start=seq_len_start,
            seq_len_end=seq_len_end,
            device=device,
            save_path=f"trained/{model_name}",
            start_epoch=start_epoch,
            use_amp=get_config_value(config, "use_amp", False),
            log_gpu_memory=get_config_value(config, "log_gpu_memory", False),
            num_workers=get_config_value(config, "num_workers", 0),
            pin_memory=get_config_value(config, "pin_memory", True),
            geometric_ratio=geometric_ratio,
            anchor_ratio=anchor_ratio,
            sigmoid_scale_start=sigmoid_scale_start,
            sigmoid_scale_end=sigmoid_scale_end,
            early_stop_patience=early_stop_patience,
        )

        print(f"Training completed successfully!")
        print(f"Model saved to: trained/{model_name}.pt")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
