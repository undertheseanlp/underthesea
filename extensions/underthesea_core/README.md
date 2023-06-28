# Underthesea Core

Underthesea Core is a powerful extension of the popular natural language processing library Underthesea, which includes a range of efficient data preprocessing tools and machine learning models for training. Built with Rust for optimal performance, Underthesea Core offers fast processing speeds and is easy to implement, with Python bindings for seamless integration into existing projects. This extension is an essential tool for developers looking to build high-performance NLP systems that deliver accurate and reliable results.

## Usage

CRFFeaturizer

```python
>>> from underthesea_core import CRFFeaturizer
>>> features = ["T[-1]", "T[0]", "T[1]"]
>>> dictionary = set(["sinh viên"])
>>> featurizer = CRFFeaturizer(features, dictionary)
>>> sentences = [[["sinh", "X"], ["viên", "X"], ["đi", "X"], ["học", "X"]]]
>>> featurizer.process(sentences)
[[['T[-1]=BOS', 'T[0]=sinh', 'T[1]=viên'],
  ['T[-1]=sinh', 'T[0]=viên', 'T[1]=đi'],
  ['T[-1]=viên', 'T[0]=đi', 'T[1]=học'],
  ['T[-1]=đi', 'T[0]=học', 'T[1]=EOS']]]
```

## Release Workflow

1. Change version in `Cargo.toml` and `pyproject.toml`
2. Push to branch `core` with commit `Publish Underthesea Core`
  * This will trigger `release-pypi-core` action
3. Check latest version in [pypi](https://pypi.org/project/underthesea_core/)

Note*: Run a self-hosted for building `macos-arm`