#!/usr/bin/env python3
"""
RWKV Training Script

Trains an RWKVModel using the same TOML-driven pipeline as agi2_train.py.
Usage: python rwkv_train.py <config_file>
"""

from agi2_train import main
from src.rwkv import RWKVModel

if __name__ == "__main__":
    main(model_cls=RWKVModel)
