# Contributing Guide

Thank you for your interest in contributing to Underthesea! This guide will help you get started.

## Types of Contributions

### Bug Reports

- Check existing [GitHub Issues](https://github.com/undertheseanlp/underthesea/issues) first
- Include Python version, OS, and Underthesea version
- Provide minimal code to reproduce the issue
- Include full error traceback

### Bug Fixes

- Reference the issue number in your PR
- Include tests that demonstrate the fix
- Update documentation if needed

### New Features

- Open an issue to discuss the feature first
- Follow the existing code style
- Add tests and documentation

### Documentation

- Fix typos and improve clarity
- Add examples and tutorials
- Translate documentation

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- uv (recommended) or pip

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/undertheseanlp/underthesea.git
cd underthesea

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e ".[dev]"
```

### macOS ARM64 (Apple Silicon)

Build the Rust extension:

```bash
cd extensions/underthesea_core
uv pip install maturin
maturin develop
cd ../..
```

## Code Style

### Linting

We use Ruff for linting:

```bash
# Check for issues
ruff check underthesea/

# Auto-fix issues
ruff check underthesea/ --fix
```

### Configuration

Ruff configuration is in `pyproject.toml`.

## Testing

### Test Categories

| Command | Description |
|---------|-------------|
| `tox -e lint` | Linting with Ruff |
| `tox -e core` | Core module tests |
| `tox -e deep` | Deep learning tests |
| `tox -e prompt` | Prompt model tests |
| `tox -e langdetect` | Language detection tests |

### Running Specific Tests

```bash
# Word tokenization tests
uv run python -m unittest discover tests.pipeline.word_tokenize

# POS tagging tests
uv run python -m unittest discover tests.pipeline.pos_tag

# NER tests
uv run python -m unittest tests.pipeline.ner.test_ner

# Classification tests
uv run python -m unittest tests.pipeline.classification.test_bank

# Translation tests
uv run python -m unittest discover tests.pipeline.translate
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure
- Use Python's `unittest` framework
- Include both positive and edge case tests

```python
import unittest
from underthesea import word_tokenize

class TestWordTokenize(unittest.TestCase):
    def test_basic(self):
        result = word_tokenize("Xin chào Việt Nam")
        self.assertEqual(result, ['Xin', 'chào', 'Việt Nam'])

    def test_empty_string(self):
        result = word_tokenize("")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
```

## Pull Request Process

### Before Submitting

1. **Update your branch**: Rebase on latest `main`
2. **Run linting**: `ruff check underthesea/`
3. **Run tests**: `tox -e core`
4. **Update docs**: If adding features

### PR Guidelines

- Use clear, descriptive titles
- Reference related issues
- Describe changes and motivation
- Include test results
- Add screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Linting passes
- [ ] Unit tests pass
- [ ] Manual testing done

## Documentation
- [ ] Updated relevant docs
- [ ] Added docstrings
```

## Project Structure

```
underthesea/
├── underthesea/           # Main package
│   ├── pipeline/          # NLP modules
│   ├── models/            # Model implementations
│   ├── datasets/          # Built-in datasets
│   ├── corpus/            # Corpus handling
│   └── cli.py             # CLI commands
├── tests/                 # Test files
├── docs/                  # Documentation
├── extensions/            # Rust extension, apps
└── pyproject.toml         # Project configuration
```

## CLI Commands

```bash
# List available data
underthesea list-data

# List available models
underthesea list-model

# Download data
underthesea download-data VNTC
```

## Getting Help

- [GitHub Issues](https://github.com/undertheseanlp/underthesea/issues)
- [Facebook Community](https://www.facebook.com/undertheseanlp/)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers feel welcome
