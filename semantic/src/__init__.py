"""
AGI2 - AGI2 Implementation Package

This package provides a complete implementation of AGI2 model architecture
with training, generation, and interactive conversation capabilities.
"""

from .attention import MultiHeadAttention

# Tokenization and data
from .basic_tokenizer import BasicTokenizer
from .bpe_tokenizer import BPETokenizer

# Core configuration
from .config import AGI2Config

# Training and generation
from .cosine_loss import PairwiseCosineLoss
from .dataset import TextDataset

# Model components
from .embeddings import PositionEmbeddings, TokenEmbeddings
from .ffn import FeedForward
from .generation import generate_text
from .interactive import InteractivePrompt
from .model import AGI2Model
from .training import train_epoch, train_model
from .transformer import TransformerBlock

# Utilities
from .utils import *

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "AGI2Config",
    # Model components
    "TokenEmbeddings",
    "PositionEmbeddings",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "AGI2Model",
    # Tokenization and data
    "BasicTokenizer",
    "BPETokenizer",
    "TextDataset",
    # Training and generation
    "PairwiseCosineLoss",
    "train_epoch",
    "train_model",
    "generate_text",
    "InteractivePrompt",
]
