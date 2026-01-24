# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Underthesea is an open-source Vietnamese Natural Language Processing (NLP) toolkit. It provides pre-trained models for common NLP tasks: sentence segmentation, word tokenization, POS tagging, chunking, NER, text classification, sentiment analysis, dependency parsing, language detection, and text-to-speech.

## Build & Development Commands

**Note:** Use `uv` for all Python/pip commands.

```bash
# Setup virtual environment
uv venv
source .venv/bin/activate

# Installation
uv pip install -e "."               # Basic install
uv pip install -e ".[deep]"         # With deep learning (torch, transformers)
uv pip install -e ".[voice]"        # With text-to-speech (jax, haiku)
uv pip install -e ".[prompt]"       # With prompt models (openai)
uv pip install -e ".[langdetect]"   # With language detection (fasttext)
uv pip install -e ".[dev]"          # With dev tools (ruff, tox)

# Build underthesea_core (Rust extension, required for macOS ARM64)
cd extensions/underthesea_core && uv pip install maturin && maturin develop && cd ../..

# Linting
ruff check underthesea/             # Fast Python linter (config in pyproject.toml)
ruff check underthesea/ --fix       # Auto-fix issues

# Run all tests by category
tox -e lint                         # Linting with ruff
tox -e core                         # Core module tests
tox -e deep                         # Deep learning tests
tox -e prompt                       # Prompt model tests (requires OPENAI_API_KEY)
tox -e langdetect                   # Language detection tests

# Run individual test modules
uv run python -m unittest discover tests.pipeline.word_tokenize
uv run python -m unittest discover tests.pipeline.pos_tag
uv run python -m unittest tests.pipeline.ner.test_ner
uv run python -m unittest tests.pipeline.classification.test_bank
uv run python -m unittest discover tests.pipeline.translate

# CLI commands
underthesea list-data               # List available datasets
underthesea list-model              # List available models
underthesea download-data VNTC      # Download a dataset
```

## Architecture

### Core Package Structure (`underthesea/`)

- **`pipeline/`** - Main NLP modules, each in its own subdirectory:
  - `sent_tokenize/` - Sentence segmentation
  - `text_normalize/` - Vietnamese text normalization
  - `word_tokenize/` - Word segmentation (CRF-based)
  - `pos_tag/` - Part-of-speech tagging
  - `chunking/` - Phrase chunking
  - `ner/` - Named entity recognition (supports `deep=True` for transformer models)
  - `classification/` - Text classification (supports `domain='bank'` and `model='prompt'`)
  - `sentiment/` - Sentiment analysis (supports `domain='bank'`)
  - `dependency_parse/` - Dependency parsing (requires `[deep]` install)
  - `lang_detect/` - Language detection (requires `[langdetect]` install)
  - `translate/` - Vietnamese-English translation (requires `[deep]` install)
  - `say/` - Text-to-speech (requires `[voice]` install)

- **`models/`** - Model implementations (CRF taggers, classifiers)
- **`datasets/`** - Built-in datasets and stopwords
- **`corpus/`** - Corpus handling and validation
- **`cli.py`** - Click-based CLI (`underthesea` command)

### Key Patterns

1. **Lazy Model Loading**: Models are loaded on first use and cached in module-level globals
2. **Optional Dependencies**: Features like deep learning, prompt models, and langdetect are behind optional installs to keep base package lightweight
3. **API Consistency**: Each pipeline module exposes a main function matching the module name (e.g., `word_tokenize()`, `ner()`, `classify()`)

### Extensions (`extensions/`)

- `underthesea_core/` - Rust-based performance module (built with maturin)
- `apps/` - Web applications (Django backends)
- `datasets/` - Additional language resources
- `labs/` - Experimental modules

## Testing

Tests use Python's built-in `unittest` framework. Test files mirror the source structure under `tests/`.

## Git Commits

Commits must have only one author and committer: `Vu Anh <anhv.ict91@gmail.com>`. Do not add any co-authors.

## Python Version

Requires Python 3.10+ (supports 3.10, 3.11, 3.12, 3.13, 3.14).
