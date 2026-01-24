# Installation

## Requirements

- Python 3.9 or higher (supports 3.9, 3.10, 3.11, 3.12, 3.13)
- pip package manager

## Basic Installation

To install Underthesea, simply run:

```bash
pip install underthesea
```

This installs the core package with basic NLP features:

- Sentence segmentation
- Text normalization
- Word segmentation
- POS tagging
- Chunking
- Named entity recognition (CRF model)
- Text classification
- Sentiment analysis

## Optional Dependencies

Underthesea provides optional dependencies for advanced features:

### Deep Learning Models

For deep learning-based features (dependency parsing, deep NER, translation):

```bash
pip install "underthesea[deep]"
```

This includes PyTorch and Transformers libraries.

### Text-to-Speech

For Vietnamese text-to-speech synthesis:

```bash
pip install "underthesea[voice]"
underthesea download-model VIET_TTS_V0_4_1
```

This includes JAX, Haiku, and audio processing libraries.

### Prompt-based Models

For OpenAI-powered text classification:

```bash
pip install "underthesea[prompt]"
```

!!! note
    You need to set the `OPENAI_API_KEY` environment variable to use prompt-based models.

### Language Detection

For language detection using FastText:

```bash
pip install "underthesea[langdetect]"
```

### Development Tools

For development and testing:

```bash
pip install "underthesea[dev]"
```

### Install All Optional Dependencies

```bash
pip install "underthesea[deep,voice,prompt,langdetect]"
```

## Installation from Source

### Clone the Repository

```bash
git clone https://github.com/undertheseanlp/underthesea.git
cd underthesea
```

### Using uv (Recommended)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e "."

# With optional dependencies
uv pip install -e ".[deep]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e "."
```

## macOS ARM64 (Apple Silicon)

For macOS with Apple Silicon (M1/M2/M3), you need to build the Rust extension:

```bash
cd extensions/underthesea_core
uv pip install maturin
maturin develop
cd ../..
```

## Verify Installation

```python
>>> import underthesea
>>> underthesea.__version__
'9.0.0'

>>> from underthesea import word_tokenize
>>> word_tokenize("Xin chào Việt Nam")
['Xin', 'chào', 'Việt Nam']
```

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you have the correct optional dependencies installed:

```bash
# For deep learning features
pip install "underthesea[deep]"

# For text-to-speech
pip install "underthesea[voice]"

# For language detection
pip install "underthesea[langdetect]"
```

### Model Download Issues

Some models are downloaded automatically on first use. If you have network issues, you can manually download models:

```bash
underthesea download-model MODEL_NAME
```

### Getting Help

- [GitHub Issues](https://github.com/undertheseanlp/underthesea/issues)
- [Facebook Community](https://www.facebook.com/undertheseanlp/)
