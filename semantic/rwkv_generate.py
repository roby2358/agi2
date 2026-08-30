#!/usr/bin/env python3
"""
RWKV Text Generation Script

Usage: uv run python rwkv_generate.py <config_file> [prompt]
"""

from agi2_generate import main
from src.rwkv import RWKVModel

if __name__ == "__main__":
    main(model_cls=RWKVModel)
