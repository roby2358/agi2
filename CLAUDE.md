# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGI2 is a complete GPT-2 implementation in PyTorch for training custom language models from text data. It covers the full pipeline: tokenization, training, checkpoint management, and text generation (sampling + beam search). All runtime configuration is done via TOML files, not CLI arguments.

## Commands

### Setup
```bash
uv sync                      # Install all dependencies
```

### Running
```bash
python agi2_train.py resources/moby_dick.toml              # Train a model
python agi2_generate.py resources/moby_dick.toml "prompt"   # Generate text
python agi2_generate_beam.py resources/moby_dick.toml "prompt"  # Beam search generation
python agi2_interactive.py resources/moby_dick.toml          # Interactive mode
python estimate_memory.py resources/lilwill.toml             # GPU memory estimation
```

### Testing
```bash
uv run pytest                                    # All tests
uv run pytest tests/test_config.py               # Single test file
uv run pytest tests/test_config.py::TestAGI2Config::test_default_config  # Single test
uv run pytest -m unit                            # Unit tests only
uv run pytest -m integration                     # Integration tests only
uv run pytest --cov=src                          # With coverage
```

### Code Quality
```bash
uv run black src/ tests/     # Format
uv run isort src/ tests/     # Sort imports
uv run flake8 src/ tests/    # Lint
uv run mypy src/             # Type check (strict mode)
```

## Architecture

The model is built bottom-up from composable `nn.Module` components in `src/`:

```
config.py          → AGI2Config (hyperparameters, presets via from_preset())
embeddings.py      → TokenEmbeddings, PositionEmbeddings
attention.py       → MultiHeadAttention
ffn.py             → FeedForward
transformer.py     → TransformerBlock (attention + FFN + layer norms)
model.py           → AGI2Model (complete GPT-2, integrates all above)
```

Supporting modules:
- `basic_tokenizer.py` / `bpe_tokenizer.py` — Character-level and BPE tokenization
- `dataset.py` — `TextDataset` for batching and sequence creation
- `training.py` — Training loop with AMP, gradient clipping, checkpointing
- `generation.py` — Text generation with temperature/top-k/top-p sampling
- `config_loader.py` — TOML config file loading and validation
- `cuda_utils.py` — Device management
- `utils.py` — Parameter counting, memory estimation, checkpoint I/O

Entry points are the top-level `agi2_*.py` scripts which load a TOML config and call into `src/`.

## Key Conventions

- **TOML-driven configuration**: Scripts take a TOML file path as their argument. Model size, training params, corpus paths — all in TOML. See `resources/default.toml` for all available parameters.
- **Requirements in SPEC.md**: Features are specified in MUST/SHOULD/MAY format in `SPEC.md`.
- **Strict typing**: mypy strict mode is enforced (`disallow_untyped_defs`, etc.). torch imports are excluded.
- **Formatting**: black (line-length 88), isort (profile "black").
- **Test markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.gpu`, `@pytest.mark.slow`.
- **Tests mirror src/**: Each `src/foo.py` has a corresponding `tests/test_foo.py`.
- **Python 3.11+** required, <3.13.
