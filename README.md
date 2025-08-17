# AGI2 - Working GPT-2 Model

A complete implementation of GPT-2 for training custom language models from text data. This project provides a full pipeline from data preparation to interactive text generation.

## Features

- **Complete GPT-2 Architecture**: Full transformer implementation with attention mechanisms
- **Custom Tokenizer**: Flexible text tokenization with vocabulary management
- **Training Pipeline**: Comprehensive training loop with gradient clipping and progress tracking
- **Text Generation**: Multiple generation strategies (sampling, beam search)
- **Interactive Mode**: Chat-like interface for testing trained models
- **Modular Design**: Clean, testable code structure

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
└── utils.py         # Utility functions
```

## Installation

### Prerequisites

- Python 3.11 or higher (3.12 recommended)
- PyTorch 2.0.0 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agi2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For development dependencies:
```bash
pip install -e ".[dev]"
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

The training module provides flexible training options. Here's how to train a model:

```python
from src.model import GPT2Model
from src.tokenizer import Tokenizer
from src.training import train_model
from src.config import ModelConfig

# Initialize model and tokenizer
config = ModelConfig(
    vocab_size=50000,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

model = GPT2Model(config)
tokenizer = Tokenizer()

# Train the model
training_history = train_model(
    model=model,
    corpus_path="path/to/your/corpus.txt",
    epochs=10,
    batch_size=4,
    learning_rate=3e-4,
    seq_len=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="trained_model.pth"
)
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

## Text Generation

### Basic Generation

Generate text from a trained model:

```python
from src.generation import generate_text
from src.tokenizer import Tokenizer

# Load your trained model
model = torch.load("trained_model.pth")
tokenizer = Tokenizer()

# Generate text
generated_text = generate_text(
    model=model,
    prompt="Once upon a time",
    max_length=100,
    temperature=0.8,
    tokenizer=tokenizer
)

print(generated_text)
```

### Generation Parameters

- **temperature**: Controls randomness (0.1 = very focused, 1.0 = very random)
- **top_k**: Limits vocabulary to top K most likely tokens
- **top_p**: Nucleus sampling - keeps tokens with cumulative probability ≤ p
- **max_length**: Maximum number of tokens to generate

### Advanced Generation

Use beam search for more coherent text:

```python
from src.generation import generate_with_beam_search

generated_text = generate_with_beam_search(
    model=model,
    prompt="The future of AI is",
    max_length=50,
    beam_size=5,
    tokenizer=tokenizer
)
```

## Interactive Mode

### Chat Interface

Use the interactive mode for testing your model:

```python
from src.interactive import InteractivePrompt

# Create interactive session
chat = InteractivePrompt(
    model=model,
    max_context_length=1024,
    tokenizer=tokenizer
)

# Start chatting
response = chat.send_message("Hello, how are you?")
print(response)

# Continue conversation
response = chat.send_message("Tell me a story")
print(response)
```

### Interactive Features

- **Context Management**: Maintains conversation history
- **Memory Efficient**: Automatically manages context length
- **Role-based**: Distinguishes between user and assistant messages

## Model Configuration

### Customizing Model Size

Adjust the model architecture in `src/config.py`:

```python
from src.config import ModelConfig

# Small model (faster training, less memory)
small_config = ModelConfig(
    vocab_size=30000,
    n_positions=512,
    n_embd=384,
    n_layer=6,
    n_head=6
)

# Large model (better quality, more memory)
large_config = ModelConfig(
    vocab_size=50000,
    n_positions=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16
)
```

## Testing

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest -m unit      # Unit tests only
pytest -m integration  # Integration tests only
```

## Performance Optimization

### GPU Acceleration

- Install CUDA-enabled PyTorch for GPU training
- Use `device="cuda"` in training functions
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

### Debug Mode

Enable verbose logging during training:

```python
training_history = train_model(
    # ... other parameters ...
    verbose=True  # Enable detailed progress output
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This implementation is based on the GPT-2 architecture described in "Language Models are Unsupervised Multitask Learners" by Radford et al.
