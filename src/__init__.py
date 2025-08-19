"""
AGI2 - AGI2 Implementation Package

This package provides a complete implementation of AGI2 model architecture
with training, generation, and interactive conversation capabilities.
"""

# Core configuration
from .config import AGI2Config

# Model components
from .embeddings import TokenEmbeddings, PositionEmbeddings
from .attention import MultiHeadAttention
from .ffn import FeedForward
from .transformer import TransformerBlock
from .model import AGI2Model

# Tokenization and data
from .basic_tokenizer import BasicTokenizer
from .bpe_tokenizer import BPETokenizer
from .dataset import TextDataset

# Training and generation
from .training import train_epoch, train_model
from .generation import generate_text
from .interactive import InteractivePrompt

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
    "train_epoch",
    "train_model",
    "generate_text",
    "InteractivePrompt",
]
