# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGI2 is a repository of small language-model experiments in PyTorch. Each experiment lives in its own top-level directory:

- `semantic/` — a from-scratch GPT-2 and an RWKV model, trained on text with cosine-similarity objectives (`pairwise`, `infonce`) or plain cross-entropy (`ce`). Full pipeline: tokenization, training, checkpoints, generation (sampling + beam search). Configured by TOML files, not CLI arguments.
- `hegel/` — a transformer compiled from a Markov chain with zero training. See `hegel/PRODUCT.md` and `hegel/CLAUDE.md`.

Shared at the root: `samples/` (corpora), `pyproject.toml` and `uv.lock` (one environment for everything), `pjpd/` (backlog). The root `justfile` holds the semantic recipes; `hegel/` has its own.

## Commands

### Setup
```bash
uv sync                      # Install all dependencies
```

The repo is shared between the host and a container, and each has its own venv: the host uses `.venv`, the container uses `.venv-container` (its `~/.bashrc` exports `UV_PROJECT_ENVIRONMENT=.venv-container`). Both are gitignored. Always go through `uv run` or `just`, never a hard-coded `.venv/bin/python`.

### Running (semantic)

From anywhere in the repo except `hegel/`, via the root justfile. The model name is a TOML basename in `semantic/resources/`; names containing "rwkv" route to the RWKV scripts.

```bash
just train lilwill                  # agi2_train.py resources/lilwill.toml
just train lilwill_rwkv             # rwkv_train.py resources/lilwill_rwkv.toml
just run lilwill_rwkv "What ho"     # generate from a trained model
```

Or directly, from `semantic/`:

```bash
cd semantic
uv run python agi2_train.py resources/moby_dick.toml
uv run python agi2_generate.py resources/moby_dick.toml "prompt"
uv run python agi2_generate_beam.py resources/moby_dick.toml "prompt"
uv run python agi2_interactive.py resources/moby_dick.toml
```

The `rwkv_*.py` scripts mirror these. `estimate_memory.py` at the root estimates GPU memory for a config.

### Running (hegel)

From `hegel/`, via its own justfile:

```bash
cd hegel
just compile                        # compile + verify: hegel_compile.py resources/corpus20.toml
just compile corpus20_squeezed      # the squeezed build
just run "What ho"                  # continue a source string; no string = interactive
```

Details, tests and lint commands are in `hegel/CLAUDE.md`.

### Testing

Run from `semantic/` (the scripts and tests import `src.*` relative to it):

```bash
cd semantic
uv run pytest                                    # All tests
uv run pytest tests/test_config.py               # Single test file
uv run pytest tests/test_config.py::TestAGI2Config::test_default_config  # Single test
uv run pytest -m unit                            # Unit tests only
uv run pytest -m integration                     # Integration tests only
uv run pytest --cov=src                          # With coverage
```

### Code Quality

```bash
cd semantic
uv run black src/ tests/     # Format
uv run isort src/ tests/     # Sort imports
uv run flake8 src/ tests/    # Lint
uv run mypy src/             # Type check (strict mode)
```

## Architecture (semantic)

The transformer is built bottom-up from composable `nn.Module` components in `semantic/src/`:

```
config.py          → AGI2Config (hyperparameters, presets via from_preset())
embeddings.py      → TokenEmbeddings, PositionEmbeddings
attention.py       → MultiHeadAttention
ffn.py             → FeedForward
transformer.py     → TransformerBlock (attention + FFN + layer norms)
model.py           → AGI2Model (complete GPT-2, integrates all above)
rwkv.py            → the RWKV model
```

Supporting modules:
- `basic_tokenizer.py` / `bpe_tokenizer.py` / `tiktoken_tokenizer.py` — tokenization
- `dataset.py` — `TextDataset` for batching and sequence creation
- `training.py` — Training loop with AMP, gradient clipping, checkpointing
- `cosine_loss.py` — the cosine-similarity training objectives
- `generation.py` — Text generation with temperature/top-k/top-p sampling
- `config_loader.py` — TOML config file loading and validation
- `model_io.py` / `utils.py` — checkpoint I/O, parameter counting, memory estimation
- `cuda_utils.py` — Device management

Entry points are the `agi2_*.py` and `rwkv_*.py` scripts in `semantic/`, which load a TOML config and call into `src/`.

## Key Conventions

- **TOML-driven configuration**: Scripts take a TOML file path as their argument. Model size, training params, corpus paths — all in TOML. See `semantic/resources/default.toml` for all available parameters.
- **Design docs per experiment**: `semantic/SPEC.md` (MUST/SHOULD/MAY requirements) and `semantic/ARCHITECTURE_LOG.md`; `hegel/PRODUCT.md`.
- **Strict typing**: mypy strict mode is enforced (`disallow_untyped_defs`, etc.). torch imports are excluded.
- **Formatting**: black (line-length 88), isort (profile "black").
- **Test markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.gpu`, `@pytest.mark.slow`.
- **Tests mirror src/**: Each `src/foo.py` has a corresponding `tests/test_foo.py`.
- **Python 3.11+** required, <3.13.
