# GPT-2 Implementation Specification

## Overview

This document specifies the requirements for implementing a working GPT-2 model trained using pairwise cosine similarity loss with frozen embeddings. The model learns to map input contexts to points in a static random embedding space, preserving the geometric relationships between token embeddings. At inference time, token selection uses cosine similarity between hidden states and the frozen embedding matrix.

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
├── config.py                # AGI2Config class and configuration management
├── config_loader.py         # TOML configuration file loader
├── embeddings.py            # TokenEmbeddings and PositionEmbeddings classes
├── attention.py             # MultiHeadAttention class (SDPA-backed)
├── ffn.py                   # FeedForward class
├── transformer.py           # TransformerBlock class
├── model.py                 # AGI2Model class
├── cosine_loss.py           # PairwiseCosineLoss class
├── basic_tokenizer.py       # BasicTokenizer class (character-level)
├── bpe_tokenizer.py         # BPETokenizer class
├── tiktoken_tokenizer.py    # TiktokenTokenizer class (GPT-2 BPE, 50k tokens)
├── dataset.py               # TextDataset class
├── training.py              # Training functions and training loop
├── generation.py            # Text generation functions (cosine similarity scoring)
├── interactive.py           # InteractivePrompt class for conversation management
├── cuda_utils.py            # CUDA availability checking and device management
└── utils.py                 # Utility functions and helpers
```

The test structure is the same under the `tests/` directory with `test_*.py` files.

### Import Structure
**MUST:**
- Use relative imports within the package (e.g., `from .config import AGI2Config`)
- Provide clean public API through `__init__.py` files
- Support both package and direct module imports

**SHOULD:**
- Group related imports logically
- Minimize circular dependencies
- Use type hints for better code documentation

### Configuration Files
**MUST:**
- Include `pyproject.toml` for project configuration
- Specify Python version requirements (3.11+, <3.13)
- Define project dependencies and development tools
- Use uv for environment management
- Include TOML configuration files in `resources/` directory
- Add `.gitignore` for version control

**Configuration Files Structure:**
```
resources/
├── default.toml             # Default configuration with all parameters
├── moby_dick.toml           # Example configuration for Moby Dick training
├── small_model.toml         # Configuration for faster training with smaller model
├── large_model.toml         # Configuration for higher quality with larger model
├── lilwill.toml             # Small Shakespeare model
├── bigwill.toml             # Larger Shakespeare model
└── rj.toml                  # Romeo & Juliet model
```

## Architecture Requirements

### Model Configuration
- **MUST**: Support configurable model size (n_layer, n_head, n_embd, n_inner)
- **MUST**: Default to GPT-2 Small configuration (12 layers, 12 heads, 768 dimensions)
- **MUST**: Support preset configurations (Small, Medium, Large)
- **SHOULD**: Support custom configuration through TOML files

### Sequence Length
- **MUST**: Support configurable sequence length
- **MUST**: Default to 1024 tokens
- **SHOULD**: Support dynamic sequence lengths up to 2048

### Frozen Embeddings
- **MUST**: Freeze token embedding weights after initialization — the embedding matrix is a static random codebook
- The transformer learns to map input contexts to fixed points in this space
- No gradient updates flow to the embedding weights during training
- This provides a stable, non-moving target for cosine similarity training

## Component Requirements

### 1. Token Embeddings (embeddings.py)
**MUST:**
- Implement `TokenEmbeddings(vocab_size, d_model, dropout_rate)` class
- Support `forward(tokens)` method returning embeddings of shape `(batch_size, seq_len, d_model)`
- Initialize weights with normal distribution (mean=0, std=0.02)
- Embedding weights MUST be frozen (`requires_grad_(False)`) after initialization

**SHOULD:**
- Support embedding dropout (configurable rate)

**Testing Requirements:**
- Test embedding dimensions and shapes
- Test weight initialization
- Test forward pass with various batch sizes and sequence lengths

### 2. Position Embeddings (embeddings.py)
**MUST:**
- Implement `PositionEmbeddings(max_seq_len, d_model)` class
- Support `forward(seq_len)` method returning position embeddings of shape `(seq_len, d_model)`
- Use sinusoidal position encoding formula

**Testing Requirements:**
- Test position embedding uniqueness
- Test sequence length handling
- Test sinusoidal encoding correctness

### 3. Multi-Head Self-Attention (attention.py)
**MUST:**
- Implement `MultiHeadAttention(d_model, n_heads, dropout)` class
- Support `forward(query, key, value, mask)` method
- Return attention output of shape `(batch_size, seq_len, d_model)`
- Use `F.scaled_dot_product_attention()` (SDPA) for automatic flash attention / memory-efficient attention selection
- Support causal masking for autoregressive generation
- Convert boolean masks (True=attend) to float additive masks for SDPA compatibility

**SHOULD:**
- Support attention dropout

**Testing Requirements:**
- Test attention mechanism with various input shapes
- Test causal masking behavior
- Test dropout application

### 4. Feed-Forward Network (ffn.py)
**MUST:**
- Implement `FeedForward(d_model, d_ff, dropout)` class
- Support `forward(x)` method
- Use GELU activation function
- Apply dropout after each linear layer

**SHOULD:**
- Support configurable activation functions

**Testing Requirements:**
- Test output dimensions
- Test GELU activation
- Test dropout application

### 5. Transformer Block (transformer.py)
**MUST:**
- Implement `TransformerBlock(d_model, n_heads, d_ff, dropout, layer_norm_epsilon)` class
- Combine attention, feed-forward, and layer normalization
- Support `forward(x, mask)` method
- Apply residual connections around each sublayer

**Testing Requirements:**
- Test residual connections
- Test layer normalization
- Test complete forward pass

### 6. AGI2 Model (model.py)
**MUST:**
- Implement `AGI2Model(config)` class taking an `AGI2Config` object
- Provide `forward(input_ids)` method returning logits of shape `(batch_size, seq_len, vocab_size)` — used for generation
- Provide `forward_hidden(input_ids)` method returning a tuple of `(logits, hidden_states)` — used for training
  - `logits`: `(batch_size, seq_len, vocab_size)`
  - `hidden_states`: `(batch_size, seq_len, n_embd)` — after final layer norm, before output projection
- Freeze token embedding weights: `self.token_embeddings.embedding.weight.requires_grad_(False)`
- Expose token embedding weights via `self.token_embeddings.embedding.weight` for similarity computation
- Cache causal masks to avoid recreation across forward passes

**SHOULD:**
- Support `torch.compile()` for fused kernels
- Allow partial forward passes for generation

**Testing Requirements:**
- Test model output shapes (both logits and hidden states)
- Test layer count configuration
- Test attention mask handling
- Test hidden states shape matches `(batch_size, seq_len, n_embd)`

### 7. Tokenization

#### BasicTokenizer (basic_tokenizer.py)
**MUST:**
- Implement `BasicTokenizer()` class
- Support `encode(text)` and `decode(tokens)` methods
- Build vocabulary from training corpus via `fit(texts)` method
- Handle basic text preprocessing

**Testing Requirements:**
- Test encode/decode roundtrip
- Test vocabulary management

#### TiktokenTokenizer (tiktoken_tokenizer.py)
**MUST:**
- Implement `TiktokenTokenizer()` class wrapping tiktoken GPT-2 BPE encoding
- Provide 50,257 subword tokens
- Support `encode(text)` and `decode(tokens)` methods
- `fit()` MUST be a no-op (vocabulary is pre-built)
- Expose `vocab_size` and `vocab` properties matching the BasicTokenizer interface

**Testing Requirements:**
- Test encode/decode roundtrip
- Test vocab size is 50,257
- Test EOS token is in vocab
- Test fit is a no-op

#### BPETokenizer (bpe_tokenizer.py)
**MUST:**
- Implement `BPETokenizer(vocab_size)` class
- Support `encode(text)` and `decode(tokens)` methods
- Train on provided text corpus

**Testing Requirements:**
- Test BPE training process
- Test encode/decode consistency

### 8. Data Loading Pipeline (dataset.py)
**MUST:**
- Implement `TextDataset(sources, tokenizer, seq_len)` class
- Support `__len__()` and `__getitem__(idx)` methods
- Produce `(prompt, single next token)` pairs — the training loop compares the last hidden vector against the target token's embedding
- Return dict with `prompt_ids` and `target_ids` tensors
- Handle file reading and text preprocessing
- Accept single path or list of paths for `sources`

**SHOULD:**
- Support data shuffling
- Support custom text preprocessing

**Testing Requirements:**
- Test dataset loading
- Test sequence length handling
- Test single-token targets
- Test multiple source files

### 9. Pairwise Cosine Similarity Loss (cosine_loss.py)
**MUST:**
- Implement `PairwiseCosineLoss(geometric_ratio, anchor_ratio)` class
- Accept two ratios: geometric (0.7) and anchor (0.3)
- Compute geometric pairs: `(sim(H_i, H_j) - sim(E_i, E_j))²` where H are hidden states and E are target embeddings from the frozen codebook
- Compute anchor pairs: `(sim(H_i, E_k) - sim(E_i, E_k))²` where E_k is a randomly sampled vocabulary embedding
- Weight losses by the configured ratios
- Exclude degenerate observations (zero-norm vectors) to avoid NaN
- Return tuple of `(total_loss, metrics_dict)`

**MUST NOT:**
- Apply any activation function (sigmoid, softmax) to hidden states before computing cosine similarity
- Include embedding pairs (unnecessary with frozen embeddings — collapse is impossible)

**SHOULD:**
- Reuse forward pass activations across pairs within a mini-batch

**Testing Requirements:**
- Test geometric pairs preserve known similarities
- Test anchor pairs compute correctly
- Test degenerate observation handling
- Test gradient flow
- Test different ratio configurations

### 10. Training Loop (training.py)
**MUST:**
- Implement `train_epoch(model, dataloader, optimizer, loss_fn, device, clip_grad_norm, scaler, log_gpu_memory)` function
- Get hidden states from model via `model.forward_hidden()`
- Use only the last hidden vector — `hidden_states[:, last_target_pos, :]` — not intermediate vectors
- Look up target embeddings from the frozen codebook for the last target token
- Compute two pair types per batch (geometric and anchor)
- Handle gradient clipping
- Support AMP (automatic mixed precision) via a `GradScaler` parameter (created once and reused across epochs)
- Support GPU memory monitoring

**MUST NOT:**
- Aggregate intermediate hidden states — only the final vector matters

**Testing Requirements:**
- Test complete training iteration with cosine loss
- Test gradient updates
- Test AMP support

### 11. Training Function (training.py)
**MUST:**
- Implement `train_model(model, tokenizer, sources, epochs, batch_size, learning_rate, seq_len, device, save_path, start_epoch, use_amp, log_gpu_memory, num_workers, pin_memory, geometric_ratio, anchor_ratio)` function
- Handle model training from start to finish using pairwise cosine similarity loss
- Save checkpoints every 5 epochs and a final model
- Return training history dict with `train_loss`, `epoch_times`, `metrics`
- Support training resumption from checkpoints
- Early stop when loss collapses to zero (< 1e-8) or plateaus for 10 epochs

**Testing Requirements:**
- Test complete training workflow
- Test checkpoint saving/loading
- Test training resumption

### 12. Text Generation (generation.py)
**MUST:**
- Implement `generate_text(model, prompt, max_length, temperature, top_k, top_p, tokenizer, device)` function
- Use cosine similarity scoring: compute `F.cosine_similarity(last_hidden_state, embedding_weight)` to produce scores over the vocabulary
- Divide scores by temperature before softmax
- Support top-k and top-p filtering
- Use `torch.inference_mode()` for generation (not `torch.no_grad()`)

**MUST NOT:**
- Use logits from the output projection for token selection — the model is trained with cosine similarity, not cross-entropy

**SHOULD:**
- Support beam search generation

**Testing Requirements:**
- Test text generation produces output
- Test temperature effects
- Test stopping conditions

### 13. Interactive Prompt System (interactive.py)
**MUST:**
- Implement `InteractivePrompt(model, max_context_length, tokenizer, temperature, top_k, top_p, device)` class
- Support `send_message(text)` method for user input
- Maintain conversation context in memory up to `max_context_length`
- Support `clear_context()` method to reset conversation history
- Return model-generated response to user input

**Testing Requirements:**
- Test context window building and management
- Test conversation flow and response generation
- Test context length limits and truncation
- Test context clearing and reset functionality

### 14. Configuration Management (config_loader.py)
**MUST:**
- Implement TOML configuration file loading
- Support validation of required configuration parameters
- Support different configuration types (training, generation, interactive)

**Testing Requirements:**
- Test configuration file loading
- Test parameter validation

### 15. CUDA Utilities (cuda_utils.py)
**MUST:**
- Implement CUDA availability checking
- Support automatic device selection
- Provide GPU memory monitoring capabilities
- Handle device fallback gracefully

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
- Log key metrics to console (geometric loss, anchor loss, valid observations)
- Support GPU memory monitoring

## Performance Requirements

### Training
**MUST:**
- Support training on CPU (minimum viable)
- Support GPU acceleration with CUDA
- Implement gradient clipping for stability
- Use SDPA (scaled dot-product attention) for automatic flash attention selection

**SHOULD:**
- Support `torch.compile()` for fused kernels
- Support AMP (automatic mixed precision) on CUDA

### Memory
**MUST:**
- Handle models up to 125M parameters on 8GB RAM
- Support configurable batch sizes for memory management

## Configuration System

### TOML Configuration Files
**MUST:**
- Support all training parameters (sources, model_name, epochs, batch_size, learning_rate, seq_len)
- Support model architecture parameters (model_positions, model_embd, model_layer, model_head, model_activation, model_dropout)
- Support cosine similarity training parameters (geometric_ratio, anchor_ratio)
- Support tokenizer selection (tokenizer: "tiktoken" or "char")
- Support performance parameters (use_compile, use_amp)
- Support device and resume options
- Support generation parameters (max_length, temperature, beam_size, model_seed)
- Support interactive parameters (max_context_length)

**Configuration Example:**
```toml
sources = ["data/corpus.txt"]
tokenizer = "tiktoken"
model_name = "my_model"
model_path = "trained/my_model.pt"

# Training
epochs = 100
batch_size = 16
learning_rate = 1e-4
seq_len = 512

# Model architecture
model_positions = 512
model_embd = 384
model_layer = 6
model_head = 6
model_activation = "gelu"
model_dropout = 0.1

# Cosine similarity training
geometric_ratio = 0.7
anchor_ratio = 0.3

# Performance
use_compile = false
use_amp = false

# Device and resume
device = "auto"
resume = ""

# Generation
max_length = 300
temperature = 0.3
beam_size = 3
model_seed = "Your seed text here"
max_context_length = 512
```

## Main Scripts

### Training Script (agi2_train.py)
**MUST:**
- Accept TOML configuration file as command line argument
- Initialize model with configuration parameters
- Freeze token embeddings after initialization
- Train model for specified epochs using cosine similarity loss
- Save trained model and checkpoints
- Support training resumption from checkpoints
- Support `torch.compile()` via `use_compile` TOML parameter

**Usage:**
```bash
uv run python agi2_train.py resources/my_config.toml
```

### Generation Script (agi2_generate.py)
**MUST:**
- Accept TOML configuration file and prompt text
- Load trained model from checkpoint
- Generate text using cosine similarity scoring
- Support temperature and sampling controls

**Usage:**
```bash
uv run python agi2_generate.py resources/my_config.toml "Your prompt here"
```

### Interactive Script (agi2_interactive.py)
**MUST:**
- Accept TOML configuration file
- Load trained model from checkpoint
- Provide interactive conversation interface
- Maintain conversation context

**Usage:**
```bash
uv run python agi2_interactive.py resources/my_config.toml
```

### Beam Search Script (agi2_generate_beam.py)
**MUST:**
- Accept TOML configuration file and prompt text
- Load trained model from checkpoint
- Generate text using beam search with cosine similarity scoring
- Support configurable beam size

**Usage:**
```bash
uv run python agi2_generate_beam.py resources/my_config.toml "Your prompt here"
```

## Development and Testing

### Environment Setup
**MUST:**
- Use uv for dependency management
- Support Python 3.11+, <3.13
- Include comprehensive test suite

**Setup Commands:**
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run isort src/ tests/

# Lint and type check
uv run flake8 src/ tests/
uv run mypy src/
```

### Testing Requirements
**MUST:**
- Test all core components
- Test all configuration file formats
- Test training resumption functionality
- Test GPU and CPU execution paths

**SHOULD:**
- Include integration tests
- Test memory management
- Test error handling scenarios
