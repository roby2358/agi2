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
├── __init__.py              # Package initialization with clean public API
├── config.py                # GPT2Config class and configuration management
├── config_loader.py         # TOML configuration file loader
├── embeddings.py            # TokenEmbeddings and PositionEmbeddings classes
├── attention.py             # MultiHeadAttention class
├── ffn.py                   # FeedForward class
├── transformer.py           # TransformerBlock class
├── model.py                 # GPT2Model class
├── tokenizer.py             # BasicTokenizer and BPETokenizer classes
├── dataset.py               # TextDataset class
├── training.py              # Training functions and training loop
├── generation.py            # Text generation functions (sampling + beam search)
├── interactive.py           # InteractivePrompt class for conversation management
├── cuda_utils.py            # CUDA availability checking and device management
└── utils.py                 # Utility functions and helpers
```

The test structure is the same under the test/ directory with test_*.py files.

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
- Specify Python version requirements (3.11+)
- Define project dependencies and development tools
- Use uv for environment management
- Include TOML configuration files in `resources/` directory
- Add `.gitignore` for version control
- Include basic documentation files

**Configuration Files Structure:**
```
resources/
├── default.toml             # Default configuration with all parameters
├── moby_dick.toml          # Example configuration for Moby Dick training
├── small_model.toml         # Configuration for faster training with smaller model
└── large_model.toml         # Configuration for higher quality with larger model
```

## Architecture Requirements

### Model Configuration
- **MUST**: Support configurable model size (n_layers, n_heads, d_model, d_ff)
- **MUST**: Default to GPT-2 Small configuration (12 layers, 12 heads, 768 dimensions)
- **MUST**: Support preset configurations (Small, Medium, Large)
- **SHOULD**: Support custom configuration through TOML files
- **MAY**: Support dynamic configuration changes at runtime

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
- Build vocabulary from training corpus

**SHOULD:**
- Support custom vocabulary
- Handle special tokens
- Support vocabulary persistence

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
- Implement `TextDataset(sources: str | list[str], tokenizer, seq_len: int)` class
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
- Implement `train_epoch(model, dataloader, optimizer, criterion, device, clip_grad_norm)` function
- Support forward pass, loss calculation, and backpropagation
- Return training loss for the epoch
- Handle gradient clipping
- Support GPU memory monitoring

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
- Implement `train_model(model, tokenizer, sources, epochs, **kwargs)` function
- Handle model training from start to finish
- Save checkpoints periodically
- Return training history
- Support training resumption from checkpoints

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
- Implement `generate_text(model, prompt, max_length, temperature, top_k, top_p, tokenizer, device)` function
- Support autoregressive text generation
- Return generated text as string
- Handle generation stopping conditions
- Support top-k and top-p sampling

**SHOULD:**
- Support beam search generation
- Implement repetition penalty
- Support custom stopping criteria

**Testing Requirements:**
- Test text generation quality
- Test temperature effects
- Test stopping conditions

### 13. Interactive Prompt System (interactive-prompt)
**MUST:**
- Implement `InteractivePrompt(model, max_context_length, tokenizer)` class
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

### 14. Configuration Management (config-loader)
**MUST:**
- Implement TOML configuration file loading
- Support validation of required configuration parameters
- Provide default values for optional parameters
- Support different configuration types (training, generation, interactive)

**SHOULD:**
- Support configuration file validation
- Provide helpful error messages for missing parameters

**Testing Requirements:**
- Test configuration file loading
- Test parameter validation
- Test default value handling

### 15. CUDA Utilities (cuda-utils)
**MUST:**
- Implement CUDA availability checking
- Support automatic device selection
- Provide GPU memory monitoring capabilities
- Handle device fallback gracefully

**SHOULD:**
- Support device preference configuration
- Provide memory usage information

**Testing Requirements:**
- Test CUDA detection
- Test device selection
- Test memory monitoring

## Integration Requirements

### Model Serialization
**MUST:**
- Support saving/loading model weights
- Preserve model configuration
- Handle vocabulary and tokenizer state
- Support checkpoint-based training resumption

**Testing Requirements:**
- Test model save/load roundtrip
- Test configuration preservation
- Test checkpoint loading

### Training Progress
**MUST:**
- Display training loss per epoch
- Show progress bars for long operations
- Log key metrics to console
- Support GPU memory monitoring

## Performance Requirements

### Training
**MUST:**
- Support training on CPU (minimum viable)
- Support GPU acceleration with CUDA
- Implement gradient clipping for stability

**SHOULD:**
- Support gradient accumulation for large models
- Implement efficient attention mechanisms

### Memory
**MUST:**
- Handle models up to 125M parameters on 8GB RAM
- Support configurable batch sizes for memory management

**SHOULD:**
- Support gradient checkpointing
- Implement efficient attention mechanisms

## Configuration System

### TOML Configuration Files
**MUST:**
- Support all training parameters (sources, model_name, epochs, batch_size, learning_rate, seq_len)
- Support model architecture parameters (model_positions, model_embd, model_layer, model_head)
- Support device and resume options
- Support generation parameters (max_length, temperature, beam_size, model_seed)
- Support interactive parameters (max_context_length)

**Configuration Examples:**
```toml
# Training configuration
sources = ["data/corpus.txt"]
model_name = "my_model"
epochs = 10
batch_size = 12
learning_rate = 3e-4
seq_len = 1024

# Model architecture
model_layer = 12
model_head = 12
model_embd = 768

# Device and resume
device = "auto"
resume = null

# Generation parameters
max_length = 100
temperature = 0.8
beam_size = 5
```

## Main Scripts

### Training Script (agi2_train.py)
**MUST:**
- Accept TOML configuration file as command line argument
- Initialize model with configuration parameters
- Train model for specified epochs
- Save trained model and checkpoints
- Support training resumption from checkpoints

**Usage:**
```bash
python agi2_train.py resources/my_config.toml
```

### Generation Script (agi2_generate.py)
**MUST:**
- Accept TOML configuration file and prompt text
- Load trained model from configuration
- Generate text based on prompt and parameters
- Support temperature and sampling controls

**Usage:**
```bash
python agi2_generate.py resources/my_config.toml "Your prompt here"
```

### Interactive Script (agi2_interactive.py)
**MUST:**
- Accept TOML configuration file
- Load trained model from configuration
- Provide interactive conversation interface
- Maintain conversation context

**Usage:**
```bash
python agi2_interactive.py resources/my_config.toml
```

### Beam Search Script (agi2_generate_beam.py)
**MUST:**
- Accept TOML configuration file and prompt text
- Load trained model from configuration
- Generate text using beam search algorithm
- Support configurable beam size

**Usage:**
```bash
python agi2_generate_beam.py resources/my_config.toml "Your prompt here"
```

## Usage

### Configuration-Based Usage
All AGI2 functionality is driven by TOML configuration files:

```bash
# Training
python agi2_train.py resources/my_project.toml

# Text generation
python agi2_generate.py resources/my_project.toml "Your prompt here"

# Interactive conversation
python agi2_interactive.py resources/my_project.toml

# Beam search generation
python agi2_generate_beam.py resources/my_project.toml "Your prompt here"
```

### Programmatic Usage
```python
from src.model import GPT2Model
from src.training import train_model
from src.interactive import InteractivePrompt
from src.config_loader import load_config

# Load configuration
config = load_config("resources/my_project.toml")

# Initialize model
model = GPT2Model(config)

# Train model
train_model(model, tokenizer, **config)

# Generate text
from src.generation import generate_text
output = generate_text(model, "Hello world", max_length=50, tokenizer=tokenizer)

# Interactive conversation
prompt = InteractivePrompt(model, max_context_length=1024, tokenizer=tokenizer)
response = prompt.send_message("Hello, how are you today?")
print(response)

# Continue conversation (context builds automatically)
response2 = prompt.send_message("What did you just say?")
print(response2)

# Clear context if needed
prompt.clear_context()
```

## Development and Testing

### Environment Setup
**MUST:**
- Use uv for dependency management
- Support Python 3.11+
- Include comprehensive test suite
- Support development dependencies

**Setup Commands:**
```bash
# Install with development dependencies
uv sync --extra=dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run isort src/ tests/
```

### Testing Requirements
**MUST:**
- Maintain 100% test coverage for core functionality
- Test all configuration file formats
- Test training resumption functionality
- Test GPU and CPU execution paths

**SHOULD:**
- Include integration tests
- Test memory management
- Test error handling scenarios
