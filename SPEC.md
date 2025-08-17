# GPT-2 Implementation Specification

## Overview

This document specifies the requirements for implementing a working GPT-2 model that can be trained on text data and generate text output. The implementation follows a bottom-up approach, building core components first, then integrating them into a complete training and generation system.

## Project Structure Requirements

### Directory Organization
**MUST:**
- Organize source code in `src/` directory
- Organize tests in `tests/` directory
- Mirror source structure in test directory
- Use `__init__.py` files for proper Python package structure
- Group related functionality in logical modules
- Maintain clear separation between core components and utilities
- Use consistent naming conventions across all files

**SHOULD:**
- Favor classes and object oriented design
- Favor functional construction on objects

### Source Code Structure (`src/`)
**MUST:**
```
src/
├── __init__.py              # Package initialization
├── config.py                # GPT2Config class and configuration management
├── embeddings.py            # TokenEmbeddings and PositionEmbeddings classes
├── attention.py             # MultiHeadAttention class
├── ffn.py                   # FeedForward class
├── transformer.py           # TransformerBlock class
├── model.py                 # GPT2Model class
├── tokenizer.py             # BasicTokenizer and BPETokenizer classes
├── dataset.py               # TextDataset class
├── training.py              # Training functions and training loop
├── generation.py            # Text generation functions
├── interactive.py            # InteractivePrompt class for conversation management
└── utils.py                 # Utility functions and helpers
```

### Test Structure (`tests/`)
**MUST:**
```
tests/
├── __init__.py              # Test package initialization
├── test_config.py           # Tests for config.py
├── test_embeddings.py       # Tests for embeddings.py
├── test_attention.py        # Tests for attention.py
├── test_ffn.py             # Tests for ffn.py
├── test_transformer.py      # Tests for transformer.py
├── test_model.py            # Tests for model.py
├── test_tokenizer.py        # Tests for tokenizer.py
├── test_dataset.py          # Tests for dataset.py
├── test_training.py         # Tests for training.py
├── test_generation.py       # Tests for generation.py
├── test_interactive.py      # Tests for interactive.py
└── test_utils.py            # Tests for utils.py
```

### Import Structure
**MUST:**
- Use relative imports within the package (e.g., `from .config import GPT2Config`)
- Provide clean public API through `__init__.py` files
- Support both package and direct module imports

**SHOULD:**
- Group related imports logically
- Minimize circular dependencies
- Use type hints for better code documentation

### Configuration Files
**MUST:**
- Include `pyproject.toml` for project configuration
- Specify Python version requirements (3.12+)
- Define project dependencies and development tools
- Use uv for environment management
- Include `requirements.txt` for easy dependency installation
- Add `.gitignore` for version control
- Include basic documentation files

## Architecture Requirements

### Model Configuration
- **MUST**: Support configurable model size (n_layers, n_heads, d_model, d_ff)
- **MUST**: Default to GPT-2 Small configuration (12 layers, 12 heads, 768 dimensions)
- **SHOULD**: Refresh configuration changes at application start
- **MAY**: Support multiple preset configurations (Small, Medium, Large)

### Sequence Length
- **MUST**: Support configurable sequence length
- **MUST**: Default to 1024 tokens
- **SHOULD**: Support dynamic sequence lengths up to 2048
- **MAY**: Support variable-length sequences with padding

## Component Requirements

### 1. Token Embeddings (embeddings-hvqe)
**MUST:**
- Implement `TokenEmbeddings(vocab_size: int, d_model: int)` class
- Support `forward(tokens: torch.Tensor) -> torch.Tensor` method
- Return embeddings of shape `(batch_size, seq_len, d_model)`
- Initialize weights with normal distribution (mean=0, std=0.02)

**SHOULD:**
- Support embedding dropout (configurable rate)
- Allow weight tying with output layer

**MAY:**
- Support learned position embeddings
- Add layer normalization

**Testing Requirements:**
- Test embedding dimensions and shapes
- Test weight initialization
- Test forward pass with various batch sizes and sequence lengths

### 2. Position Embeddings (embeddings-hvqe)
**MUST:**
- Implement `PositionEmbeddings(max_seq_len: int, d_model: int)` class
- Support `forward(seq_len: int) -> torch.Tensor` method
- Return position embeddings of shape `(seq_len, d_model)`
- Use sinusoidal position encoding formula

**SHOULD:**
- Support learned position embeddings as alternative
- Allow max sequence length configuration

**Testing Requirements:**
- Test position embedding uniqueness
- Test sequence length handling
- Test sinusoidal encoding correctness

### 3. Multi-Head Self-Attention (attention-d5cv)
**MUST:**
- Implement `MultiHeadAttention(d_model: int, n_heads: int, dropout: float = 0.1)` class
- Support `forward(query, key, value, mask=None)` method
- Return attention output of shape `(batch_size, seq_len, d_model)`
- Implement scaled dot-product attention mechanism
- Support causal masking for autoregressive generation

**SHOULD:**
- Support attention dropout
- Allow custom attention masks
- Support key/value caching for generation

**Testing Requirements:**
- Test attention mechanism with various input shapes
- Test causal masking behavior
- Test dropout application
- Test attention weights sum to 1

### 4. Feed-Forward Network (ffn-mw74)
**MUST:**
- Implement `FeedForward(d_model: int, d_ff: int, dropout: float = 0.1)` class
- Support `forward(x: torch.Tensor) -> torch.Tensor` method
- Use GELU activation function
- Apply dropout after each linear layer

**SHOULD:**
- Support configurable activation functions
- Allow bias configuration

**Testing Requirements:**
- Test output dimensions
- Test GELU activation
- Test dropout application

### 5. Transformer Block (model-aaaa)
**MUST:**
- Implement `TransformerBlock(d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1)` class
- Combine attention, feed-forward, and layer normalization
- Support `forward(x: torch.Tensor, mask=None)` method
- Apply residual connections around each sublayer

**SHOULD:**
- Support pre-norm vs post-norm configurations
- Allow dropout rate configuration

**Testing Requirements:**
- Test residual connections
- Test layer normalization
- Test complete forward pass

### 6. GPT-2 Model (model-aaaa)
**MUST:**
- Implement `GPT2Model(config: GPT2Config)` class
- Support `forward(input_ids: torch.Tensor, attention_mask=None)` method
- Return logits of shape `(batch_size, seq_len, vocab_size)`
- Support configurable number of layers

**SHOULD:**
- Support gradient checkpointing for memory efficiency
- Allow partial forward passes for generation

**Testing Requirements:**
- Test model output shapes
- Test layer count configuration
- Test attention mask handling

### 7. Basic Tokenization (tokenizer-aaac)
**MUST:**
- Implement `BasicTokenizer()` class
- Support `encode(text: str) -> List[int]` method
- Support `decode(tokens: List[int]) -> str` method
- Handle basic text preprocessing (lowercase, whitespace)

**SHOULD:**
- Support custom vocabulary
- Handle special tokens

**Testing Requirements:**
- Test encode/decode roundtrip
- Test special character handling
- Test vocabulary management

### 8. Byte-Pair Encoding (bpe-ys58)
**MUST:**
- Implement `BPETokenizer(vocab_size: int = 50000)` class
- Support `encode(text: str) -> List[int]` method
- Support `decode(tokens: List[int]) -> str` method
- Train on provided text corpus

**SHOULD:**
- Support vocabulary persistence
- Handle unknown tokens gracefully

**Testing Requirements:**
- Test BPE training process
- Test encode/decode consistency
- Test vocabulary size limits

### 9. Data Loading Pipeline (data-nkyd)
**MUST:**
- Implement `TextDataset(corpus_path: str, tokenizer, seq_len: int)` class
- Support `__len__()` and `__getitem__(idx)` methods
- Return tokenized sequences of specified length
- Handle file reading and text preprocessing

**SHOULD:**
- Support multiple file formats
- Implement data shuffling
- Support custom text preprocessing

**Testing Requirements:**
- Test dataset loading
- Test sequence length handling
- Test data iteration

### 10. Training Loop (training-zyhg)
**MUST:**
- Implement `train_epoch(model, dataloader, optimizer, criterion)` function
- Support forward pass, loss calculation, and backpropagation
- Return training loss for the epoch
- Handle gradient clipping

**SHOULD:**
- Support learning rate scheduling
- Implement early stopping
- Log training metrics

**Testing Requirements:**
- Test complete training iteration
- Test loss calculation
- Test gradient updates

### 11. Training Function (train-aaab)
**MUST:**
- Implement `train_model(model, corpus_path: str, epochs: int, **kwargs)` function
- Handle model training from start to finish
- Save checkpoints periodically
- Return training history

**SHOULD:**
- Support validation during training
- Implement model checkpointing
- Support training resumption

**Testing Requirements:**
- Test complete training workflow
- Test checkpoint saving/loading
- Test training resumption

### 12. Text Generation (generate-h7a3)
**MUST:**
- Implement `generate_text(model, prompt: str, max_length: int, temperature: float = 1.0)` function
- Support autoregressive text generation
- Return generated text as string
- Handle generation stopping conditions

**SHOULD:**
- Support top-k and top-p sampling
- Implement repetition penalty
- Support custom stopping criteria

**Testing Requirements:**
- Test text generation quality
- Test temperature effects
- Test stopping conditions

### 13. Interactive Prompt System (interactive-prompt)
**MUST:**
- Implement `InteractivePrompt(model, max_context_length: int = 1024)` class
- Support `send_message(text: str) -> str` method for user input
- Maintain conversation context in memory up to max_context_length
- Build context window progressively as conversation continues
- Support `clear_context()` method to reset conversation history
- Return model-generated response to user input

**Testing Requirements:**
- Test context window building and management
- Test conversation flow and response generation
- Test context length limits and truncation
- Test context clearing and reset functionality

### 14. Main Training Script (main-7dye)
**MUST:**
- Accept corpus file path as command line argument
- Initialize model with default configuration
- Train model for specified epochs
- Save trained model

**SHOULD:**
- Support configuration file input
- Implement progress logging
- Support training interruption

**Testing Requirements:**
- Test command line argument parsing
- Test complete training workflow
- Test model saving

## Integration Requirements

### Model Serialization
**MUST:**
- Support saving/loading model weights
- Preserve model configuration
- Handle vocabulary and tokenizer state

**Testing Requirements:**
- Test model save/load roundtrip
- Test configuration preservation

### Training Progress
**MUST:**
- Display training loss per epoch
- Show progress bars for long operations
- Log key metrics to console

## Performance Requirements

### Training
**MUST:**
- Support training on CPU (minimum viable)
**SHOULD:**
- Support GPU acceleration
- Implement gradient accumulation for large models

### Memory
**MUST:**
- Handle models up to 125M parameters on 8GB RAM
**SHOULD:**
- Support gradient checkpointing
- Implement efficient attention mechanisms

## Open Questions

### Core Implementation Questions:
**Model Architecture:**
- What's the minimum model size that will actually work? (12 layers? 6? 4?)
- Do we need the full GPT-2 architecture or can we simplify some components?
- What sequence length should we target? (1024? 512? 256?)

**Training:**
- What's the minimum dataset size needed to see meaningful results?
- How many epochs before we can expect coherent text generation?
- What learning rate and batch size will work without tuning?

**Tokenization:**
- Do we implement BPE from scratch or use a simple character-level approach initially?
- What vocabulary size is practical for a working model?

**Data:**
- What text format is simplest to start with? (plain text files?)
- How do we handle different text lengths and batching?

### Simplification Questions:

**What can we skip initially?**
- Do we need position embeddings or can we start without them?
- Can we skip layer normalization in early iterations?
- Do we need attention masking or can we assume fixed-length sequences?

**What's the minimum viable product?**
- Can we start with just forward pass and basic training?
- Do we need validation/evaluation metrics to know it's working?
- What's the simplest way to verify the model is learning?

*Goal: Identify what's essential for a working model vs. what's optimization. Shortest path from "hello world" to "generating coherent text".*


## Usage

```python
from src.model import GPT2Model
from src.train import train
from src.interactive import InteractivePrompt

# Load your text data
with open("your_corpus.txt", "r") as f:
    text = f.read()

# Train model
model = GPT2Model()
train(model, text, epochs=10)

# Generate text
output = model.generate("Hello world", max_length=50)
print(output)

# Interactive conversation
prompt = InteractivePrompt(model, max_context_length=1024)
response = prompt.send_message("Hello, how are you today?")
print(response)

# Continue conversation (context builds automatically)
response2 = prompt.send_message("What did you just say?")
print(response2)

# Clear context if needed
prompt.clear_context()
```
