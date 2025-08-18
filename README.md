# AGI2 - Working GPT-2 Model

A complete implementation of GPT-2 for training custom language models from text data. This project provides a full pipeline from data preparation to interactive text generation.

## Features

- **Complete GPT-2 Architecture**: Full transformer implementation with attention mechanisms
- **Custom Tokenizer**: Flexible text tokenization with vocabulary management
- **Training Pipeline**: Comprehensive training loop with gradient clipping and progress tracking
- **Text Generation**: Multiple generation strategies (sampling, beam search)
- **Interactive Mode**: Chat-like interface for testing trained models
- **Modular Design**: Clean, testable code structure
- **Configuration-Based**: TOML configuration files for easy parameter management

## Quick Start

### Prerequisites

- Python 3.11 or higher
- PyTorch 2.0.0 or higher
- CUDA (optional, for GPU acceleration)

### Development Setup

```bash
# Clone and setup
git clone <your-fork-url>
cd agi2

# Install with development dependencies
uv sync --extra=dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run isort src/ tests/
```

### Configuration-Based Usage

All AGI2 scripts use TOML configuration files instead of command line arguments. This makes it easier to manage different training configurations and ensures reproducibility.

```bash
# Training with configuration file
python agi2_train.py resources/moby_dick.toml

# Text generation with configuration file
python agi2_generate.py resources/moby_dick.toml " Over the starboard we saw "

# Interactive chat with configuration file
python agi2_interactive.py resources/moby_dick.toml

# Beam search generation with configuration file
python agi2_generate_beam.py resources/moby_dick.toml "The future of AI is"
```

### Configuration Files

Configuration files are stored in the `resources/` directory. Each file specifies all parameters for a specific use case:

- `resources/default.toml` - Default configuration with all parameters
- `resources/moby_dick.toml` - Example configuration for Moby Dick training
- `resources/small_model.toml` - Configuration for faster training with smaller model
- `resources/large_model.toml` - Configuration for higher quality with larger model
- Create your own `.toml` files for different projects

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest -m unit      # Unit tests only
uv run pytest -m integration  # Integration tests only
```

## Configuration Management

### All Available Parameters

The following parameters can be configured in your TOML files:

#### Training Parameters (agi2_train.py)
- `sources` - List of training data source files (required)
- `model_name` - Name for the model (required)
- `epochs` - Number of training epochs (default: 10)
- `batch_size` - Training batch size (default: 12)
- `learning_rate` - Learning rate (default: 3e-4)
- `seq_len` - Sequence length (default: 1024)
- `resume` - Path to checkpoint file to resume from (optional)
- `device` - Device to use: "cpu", "cuda", or "auto" (default: "auto")

#### Model Architecture Parameters (optional)
- `model_positions` - Maximum sequence length (default: 1024)
- `model_embd` - Embedding dimension (default: 768)
- `model_layer` - Number of transformer layers (default: 12)
- `model_head` - Number of attention heads (default: 12)

#### Generation Parameters (agi2_generate.py, agi2_generate_beam.py)
- `model_path` - Path to the trained model file (required)
- `max_length` - Maximum length of generated text (default: 100 for generate, 50 for beam)
- `temperature` - Sampling temperature for generation (default: 0.8)
- `beam_size` - Beam size for beam search (default: 5)
- `model_seed` - Multi-line text prepended to user prompts (optional)
- `device` - Device to use: "cpu", "cuda", or "auto" (default: "auto")

#### Interactive Parameters (agi2_interactive.py)
- `model_path` - Path to the trained model file (required)
- `max_context_length` - Maximum context length for chat (default: 1024)
- `device` - Device to use: "cpu", "cuda", or "auto" (default: "auto")

### Creating Custom Configuration Files

You can create custom configuration files for different projects or experiments. Simply copy one of the existing files and modify the parameters:

```bash
# Copy the default configuration
cp resources/default.toml resources/my_experiment.toml

# Edit the configuration file
# Then use it with any script
python agi2_train.py resources/my_experiment.toml
```

### Configuration File Examples

**For a small model (faster training):**
```toml
sources = ["data/small_corpus.txt"]
model_name = "small_model"
epochs = 5
batch_size = 16
model_layer = 6
model_head = 6
model_embd = 384
```

**For a large model (better quality):**
```toml
sources = ["data/large_corpus.txt"]
model_name = "large_model"
epochs = 20
batch_size = 8
model_layer = 24
model_head = 16
model_embd = 1024
```

**For GPU training:**
```toml
sources = ["data/corpus.txt"]
model_name = "gpu_model"
device = "cuda"
batch_size = 32  # Larger batch size for GPU
```

## Project Structure

```
src/
├── model.py          # GPT-2 model architecture
├── transformer.py    # Transformer blocks and attention
├── attention.py      # Multi-head attention mechanism
├── ffn.py           # Feed-forward networks
├── embeddings.py     # Token and position embeddings
├── tokenizer.py     # Text tokenization and vocabulary
├── dataset.py       # Data loading and preprocessing
├── training.py      # Training functions and loops
├── generation.py    # Text generation algorithms
├── interactive.py   # Interactive conversation system
├── config.py        # Model configuration
├── config_loader.py # TOML configuration loader
└── utils.py         # Utility functions
```

## Installation

### Using uv (Recommended)

The project uses `uv` for dependency management, which provides fast, reliable package installation:

```bash
# Install uv if you don't have it
pip install uv

# Install project dependencies
uv sync --extra=dev
```

### Manual Installation

If you prefer pip:

```bash
# Install core dependencies
pip install torch numpy tqdm rich

# Install development dependencies
pip install pytest pytest-cov pytest-xdist pytest-mock
pip install black isort flake8 mypy pre-commit
```

## Data Preparation

### Text Corpus Format

Prepare your training data as a plain text file. The model will learn from this text to generate similar content.

**Example corpus.txt:**
```
This is the first sentence of your training data.
The model will learn patterns from this text.
Make sure your corpus is large enough for good results.
Include diverse examples of the writing style you want to learn.
```

### Corpus Requirements

- **Size**: Minimum 1MB, recommended 10MB+ for good results
- **Format**: Plain text with natural line breaks
- **Quality**: Clean, well-formatted text without excessive noise
- **Encoding**: UTF-8 encoding

## Training

### Basic Training

The training module provides flexible training options. Here's how to train a model using configuration files:

```bash
# Train with a configuration file
python agi2_train.py resources/my_project.toml
```

### Training Parameters

- **epochs**: Number of complete passes through the training data
- **batch_size**: Number of sequences processed together (adjust based on memory)
- **learning_rate**: Step size for gradient updates (3e-4 is a good starting point)
- **seq_len**: Length of training sequences (1024 is standard for GPT-2)
- **device**: Training device ("cpu" or "cuda")

### Training Tips

1. **Start Small**: Begin with a smaller model (fewer layers) to test your setup
2. **Monitor Loss**: Watch the training loss - it should decrease over time
3. **Gradient Clipping**: Already implemented to prevent exploding gradients
4. **Save Checkpoints**: The training function automatically saves the model
5. **GPU Memory**: Adjust batch_size based on your GPU memory capacity

### Resuming Training

You can resume training from where you left off by setting the `resume` parameter in your configuration file:

**Configuration file with resume (resources/resume_training.toml):**
```toml
sources = ["data/my_corpus.txt"]
model_name = "my_project"
epochs = 5  # Continue for 5 more epochs
resume = "trained/my_project.pt_epoch_10.pt"  # Resume from checkpoint
device = "auto"
```

**Usage:**
```bash
# Resume training with configuration file
python agi2_train.py resources/resume_training.toml
```

**What gets resumed:**
- Model weights and parameters
- Training progress (continues from the checkpoint epoch)
- Tokenizer vocabulary (if saved in checkpoint)

**What gets reset:**
- Optimizer state (reinitialized for simplicity)
- Learning rate (uses the value from configuration file)

**Checkpoint files:**
- `trained/{model_name}.pt_epoch_{N}.pt` - Saved every 5 epochs
- `trained/{model_name}.pt` - Final model after training completes

**Resume workflow:**
1. Training stops or is interrupted
2. Use `resume` parameter in configuration file with the latest checkpoint file
3. Training continues from that epoch
4. New checkpoints are created with updated epoch numbers

## Text Generation

### Basic Generation

Generate text from a trained model using configuration files:

```bash
# Generate text with configuration file
python agi2_generate.py resources/my_project.toml "Your prompt here"
```

**Configuration file for generation (resources/generate.toml):**
```toml
model_path = "trained/my_project.pt"
max_length = 100
temperature = 0.8
model_seed = """
This is a helpful AI assistant that provides creative and informative responses.
"""
device = "auto"
```

## Performance Optimization

### GPU Acceleration

- The project automatically installs CUDA-enabled PyTorch via `uv sync`
- Use `device="cuda"` in your configuration files
- Monitor GPU memory usage and adjust batch_size accordingly

### Memory Management

- Use gradient checkpointing for large models
- Implement data streaming for very large datasets
- Consider mixed precision training for memory efficiency

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or sequence length
2. **Slow Training**: Check if using GPU, reduce model size
3. **Poor Generation**: Increase training epochs, check data quality
4. **Tokenization Errors**: Ensure corpus is properly formatted

### Environment Issues

If you encounter dependency problems:

```bash
# Clean reinstall with uv
uv venv --reinstall
uv sync --extra=dev
```

### Debug Mode

Enable verbose logging during training by modifying the training script or adding debug parameters to your configuration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass with `uv run pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation is based on the GPT-2 architecture described in "Language Models are Unsupervised Multitask Learners" by Radford et al.
