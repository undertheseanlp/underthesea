# Underthesea Core

[![PyPI version](https://badge.fury.io/py/underthesea-core.svg)](https://badge.fury.io/py/underthesea-core)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Underthesea Core is a powerful extension of the popular natural language processing library Underthesea, which includes a range of efficient data preprocessing tools and machine learning models for training. Built with Rust for optimal performance, Underthesea Core offers fast processing speeds and is easy to implement, with Python bindings for seamless integration into existing projects. This extension is an essential tool for developers looking to build high-performance NLP systems that deliver accurate and reliable results.

## Installation

```bash
pip install underthesea-core
```

## Version

**Current version: 2.0.0**

### What's New in 2.0.0

- L-BFGS optimizer with OWL-QN for L1 regularization
- 10x faster feature lookup with flat data structure
- 1.24x faster than python-crfsuite for word segmentation
- Loop unrolling and unsafe bounds-check elimination for performance

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