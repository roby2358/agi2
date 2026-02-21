# AGI2

A GPT-2 implementation in PyTorch trained with pairwise cosine similarity loss. Covers the full pipeline: tokenization, training with curriculum learning, checkpoint management, and text generation (sampling + beam search). All runtime configuration is via TOML files.

## Prerequisites

- Python 3.11+ (< 3.13)
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Setup

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies
uv sync
```

## Running

### Train a model

```bash
uv run python agi2_train.py resources/small_model.toml
```

This reads the TOML config, builds a tokenizer from the corpus, creates the model, and trains it. Checkpoints are saved every 5 epochs to `trained/`, and the final model is saved at the end.

### Generate text

```bash
uv run python agi2_generate.py resources/small_model.toml "Once upon a time"
```

### Beam search generation

```bash
uv run python agi2_generate_beam.py resources/small_model.toml "The ship sailed"
```

### Interactive mode

```bash
uv run python agi2_interactive.py resources/small_model.toml
```

### Estimate GPU memory

```bash
uv run python estimate_memory.py resources/small_model.toml
```

## Configuration

All scripts take a TOML file as their single argument. Example configs live in `resources/`. To create your own, copy one and edit it:

```bash
cp resources/small_model.toml resources/my_experiment.toml
```

### Training config example

```toml
# Data
sources = ["resources/training/corpus.txt"]
model_name = "my_model"

# Training
epochs = 50
batch_size = 12
learning_rate = 3e-4
seq_len = 512
device = "auto"

# Model architecture
model_positions = 512
model_embd = 384
model_layer = 6
model_head = 6

# Cosine similarity training
geometric_ratio = 0.5
anchor_ratio = 0.3
embedding_ratio = 0.2
curriculum_stage = 1
stage_patience = 5
position_decay = 0.5

# Generation (used by generate scripts)
model_path = "trained/my_model.pt"
max_length = 200
temperature = 0.8
beam_size = 5
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `sources` | List of text corpus file paths (required) |
| `model_name` | Name for saving the model (required) |
| `epochs` | Number of training epochs |
| `batch_size` | Sequences per batch |
| `learning_rate` | AdamW learning rate |
| `seq_len` | Maximum sequence length |
| `device` | `"cpu"`, `"cuda"`, or `"auto"` |
| `resume` | Path to checkpoint to resume from |
| `model_embd` | Embedding dimension |
| `model_layer` | Number of transformer layers |
| `model_head` | Number of attention heads |
| `geometric_ratio` | Weight for geometric pair loss |
| `anchor_ratio` | Weight for anchor pair loss |
| `embedding_ratio` | Weight for embedding pair loss |
| `curriculum_stage` | Starting curriculum stage (1, 2, or 3) |
| `stage_patience` | Epochs before advancing curriculum stage |
| `position_decay` | Exponential decay for stage 2 aggregation |
| `model_path` | Path to trained model (for generation) |
| `max_length` | Max tokens to generate |
| `temperature` | Sampling temperature |
| `beam_size` | Beam width for beam search |
| `model_seed` | Text prepended to prompts during generation |

### Provided configs

| File | Purpose |
|------|---------|
| `resources/default.toml` | Full config with all parameters |
| `resources/small_model.toml` | Small model, fast training |
| `resources/large_model.toml` | Larger model, better quality |
| `resources/moby_dick.toml` | Train on Moby Dick |

## Training approach

AGI2 uses **pairwise cosine similarity loss** instead of cross-entropy. The model learns to preserve the geometric relationships defined by the embedding matrix. The loss is:

```
(sim(A', B') - sim(A, B))^2
```

Three pair types are computed per batch:
- **Geometric**: hidden state vs hidden state similarity should match embedding similarity
- **Anchor**: hidden state vs random vocab embedding similarity should match embedding similarity
- **Embedding**: forwarded embedding vs forwarded embedding similarity should match raw embedding similarity

Training uses curriculum learning with three stages:
1. **Stage 1**: Prompt + single next token (model learns basic token mapping)
2. **Stage 2**: Prompt + 2-5 token continuation (exponentially decaying position weights)
3. **Stage 3**: Prompt + full response (arithmetic mean aggregation)

The curriculum advances automatically when loss plateaus for `stage_patience` epochs.

### Resuming training

Set `resume` in your TOML to the checkpoint path:

```toml
resume = "trained/my_model.pt_epoch_10.pt"
```

Checkpoints save model weights, optimizer state, tokenizer, and curriculum stage.

## Testing

```bash
uv run pytest                          # All tests
uv run pytest tests/test_training.py   # Single file
uv run pytest -m unit                  # Unit tests only
uv run pytest -m integration           # Integration tests only
uv run pytest --cov=src                # With coverage
```

## Code quality

```bash
uv run black src/ tests/      # Format
uv run isort src/ tests/       # Sort imports
uv run flake8 src/ tests/      # Lint
uv run mypy src/               # Type check
```

## License

MIT
