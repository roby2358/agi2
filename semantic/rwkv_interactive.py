#!/usr/bin/env python3
"""
RWKV Interactive Generation Script

Usage: uv run python rwkv_interactive.py <config_file>
"""

from agi2_interactive import main
from src.rwkv import RWKVModel

if __name__ == "__main__":
    main(model_cls=RWKVModel)
