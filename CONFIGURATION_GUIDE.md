# AGI2 Configuration Guide

This guide explains how to use the new TOML-based configuration system for all AGI2 scripts.

## Overview

All AGI2 scripts now use TOML configuration files instead of command line arguments. This provides:

- **Reproducibility**: Exact same parameters every time
- **Organization**: Different configs for different projects
- **Simplicity**: No need to remember long command lines
- **Flexibility**: Easy to experiment with different settings

## Quick Start

### Basic Usage

```bash
# Training
python agi2_train.py resources/my_config.toml

# Text generation
python agi2_generate.py resources/my_config.toml "Your prompt here"

# Interactive chat
python agi2_interactive.py resources/my_config.toml

# Beam search generation
python agi2_generate_beam.py resources/my_config.toml "Your prompt here"
```

### Example Configuration Files

The `resources/` directory contains several example configurations:

- `default.toml` - Standard configuration with all parameters
- `moby_dick.toml` - Example for Moby Dick training
- `small_model.toml` - Fast training with smaller model
- `large_model.toml` - High quality with larger model

## Configuration Parameters

### Training Parameters (agi2_train.py)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `corpus_path` | ✅ | - | Path to training corpus file |
| `model_name` | ✅ | - | Name for the model |
| `epochs` | ❌ | 10 | Number of training epochs |
| `batch_size` | ❌ | 12 | Training batch size |
| `learning_rate` | ❌ | 3e-4 | Learning rate |
| `seq_len` | ❌ | 1024 | Sequence length |
| `resume` | ❌ | null | Checkpoint file to resume from |
| `device` | ❌ | "auto" | Device: "cpu", "cuda", or "auto" |

### Model Architecture Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_positions` | ❌ | 1024 | Maximum sequence length |
| `model_embd` | ❌ | 768 | Embedding dimension |
| `model_layer` | ❌ | 12 | Number of transformer layers |
| `model_head` | ❌ | 12 | Number of attention heads |

### Generation Parameters (agi2_generate.py, agi2_generate_beam.py)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_path` | ✅ | - | Path to trained model file |
| `max_length` | ❌ | 100/50 | Maximum generated text length |
| `temperature` | ❌ | 0.8 | Sampling temperature |
| `beam_size` | ❌ | 5 | Beam size for beam search |
| `model_seed` | ❌ | "" | Multi-line text prepended to user prompts |
| `device` | ❌ | "auto" | Device: "cpu", "cuda", or "auto" |

### Interactive Parameters (agi2_interactive.py)

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_path` | ✅ | - | Path to trained model file |
| `max_context_length` | ❌ | 1024 | Maximum chat context length |
| `device` | ❌ | "auto" | Device: "cpu", "cuda", or "auto" |

## Creating Custom Configurations

### 1. Copy an Existing Configuration

```bash
cp resources/default.toml resources/my_project.toml
```

### 2. Edit the Configuration

```toml
# My Project Configuration
corpus_path = "data/my_corpus.txt"
model_name = "my_project"
epochs = 15
batch_size = 16
learning_rate = 1e-4
device = "cuda"
```

### 3. Use the Configuration

```bash
python agi2_train.py resources/my_project.toml
```

## Configuration Examples

### Fast Training (Small Model)

```toml
corpus_path = "data/corpus.txt"
model_name = "fast_model"
epochs = 5
batch_size = 16
model_layer = 6
model_head = 6
model_embd = 384
seq_len = 512
```

### High Quality (Large Model)

```toml
corpus_path = "data/large_corpus.txt"
model_name = "quality_model"
epochs = 20
batch_size = 8
model_layer = 24
model_head = 16
model_embd = 1024
seq_len = 2048
```

### GPU Training

```toml
corpus_path = "data/corpus.txt"
model_name = "gpu_model"
device = "cuda"
batch_size = 32
epochs = 10
```

### Resume Training

```toml
corpus_path = "data/corpus.txt"
model_name = "my_model"
resume = "trained/my_model.pt_epoch_10.pt"
epochs = 5
```

### Text Generation with Context

```toml
model_path = "trained/my_project.pt"
max_length = 100
temperature = 0.8
model_seed = """
This is a helpful AI assistant that provides creative and informative responses.
The assistant is knowledgeable about many topics and always tries to be helpful.
"""
device = "auto"
```

## Best Practices

### 1. Organize by Project

Create separate configuration files for different projects:

```
resources/
├── project_a.toml
├── project_b.toml
├── experiment_1.toml
└── production.toml
```

### 2. Use Descriptive Names

```toml
# Good
model_name = "shakespeare_sonnets_v1"

# Avoid
model_name = "model"
```

### 3. Document Your Changes

```toml
# Shakespeare Sonnets Training
# Trained on: 2024-01-15
# Purpose: Generate sonnet-style poetry
# Notes: Reduced learning rate for stability
corpus_path = "data/shakespeare_sonnets.txt"
model_name = "shakespeare_sonnets_v1"
learning_rate = 1e-4  # Reduced from 3e-4
```

### 4. Version Control

Keep your configuration files in version control to track experiments:

```bash
git add resources/my_experiment.toml
git commit -m "Add configuration for experiment with larger model"
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure the configuration file path is correct
2. **Missing Required Parameters**: Check that `corpus_path` and `model_name` are set
3. **Invalid TOML**: Use a TOML validator to check syntax
4. **Path Issues**: Use forward slashes or escaped backslashes in Windows paths

### Validation

The configuration loader will validate your configuration and provide helpful error messages:

```bash
python agi2_train.py resources/invalid.toml
# Error: Missing required configuration keys: ['corpus_path']
```

## Migration from Command Line

### Old Way (Command Line)

```bash
python agi2_train.py data/corpus.txt --epochs 20 --batch-size 8 --learning-rate 1e-4
```

### New Way (Configuration File)

**resources/my_config.toml:**
```toml
corpus_path = "data/corpus.txt"
epochs = 20
batch_size = 8
learning_rate = 1e-4
```

**Usage:**
```bash
python agi2_train.py resources/my_config.toml
```

## Advanced Usage

### Using Model Seeds for Context

The `model_seed` parameter allows you to provide a multi-line context that gets prepended to every user prompt. This is useful for:

- **Setting the tone and style** of responses
- **Providing background context** for the model
- **Creating character personas** or specialized assistants
- **Ensuring consistent behavior** across different prompts

**Example with Moby Dick style:**
```toml
model_seed = """
Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.
"""
```

**Example with AI assistant persona:**
```toml
model_seed = """
You are a helpful AI assistant with expertise in machine learning and artificial intelligence. You provide clear, accurate, and helpful responses. Always explain complex concepts in simple terms and provide practical examples when possible.
"""
```

**How it works:**
1. The `model_seed` text is loaded from your configuration file
2. When you run the generation script, your prompt is appended to the seed
3. The combined text is fed to the model for generation
4. This gives the model context and style guidance before processing your request

### Environment-Specific Configurations

Create different configs for different environments:

```bash
# Development
python agi2_train.py resources/dev.toml

# Testing
python agi2_train.py resources/test.toml

# Production
python agi2_train.py resources/prod.toml
```

### Conditional Parameters

Use different configurations based on your needs:

```toml
# resources/gpu.toml
device = "cuda"
batch_size = 32

# resources/cpu.toml
device = "cpu"
batch_size = 8
```

This configuration system makes AGI2 much easier to use and maintain, while providing the flexibility to experiment with different parameters and reproduce results consistently.
