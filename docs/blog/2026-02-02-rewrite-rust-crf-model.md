---
slug: rewrite-rust-crf-model
title: Rewriting CRF Model in Rust
authors: [rain1024]
tags: [rust, performance, crf, nlp]
---

# Rewriting CRF Model in Rust: A 20% Performance Boost

In underthesea v9.2.5, we completed the migration from `python-crfsuite` to our native Rust implementation `underthesea-core`. This change resulted in a **20% performance improvement** across all CRF-based NLP tasks.

<!-- truncate -->

## Background

Underthesea uses Conditional Random Fields (CRF) for several core NLP tasks:
- Word tokenization
- POS tagging
- Named Entity Recognition (NER)
- Chunking

Previously, we relied on `python-crfsuite`, a Python wrapper for the CRFsuite C++ library. While functional, this introduced overhead from multiple language boundaries.

## The Architecture Change

### Before (v9.2.1)

```
┌─────────────────────────────────────────────────────────────┐
│                        Python                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Input     │───▶│ CRFFeaturizer│───▶│   Python     │  │
│  │   Tokens     │    │    (Rust)    │    │    List      │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│                                                  ▼          │
│                                          ┌──────────────┐  │
│                                          │  pycrfsuite  │  │
│                                          │    (C++)     │  │
│                                          └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The data flow crossed multiple language boundaries:
1. Python → Rust (CRFFeaturizer)
2. Rust → Python (feature list)
3. Python → C++ (pycrfsuite)
4. C++ → Python (tags)

### After (v9.2.5)

```
┌─────────────────────────────────────────────────────────────┐
│                        Python                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Input     │───▶│ CRFFeaturizer│───▶│  CRFTagger   │  │
│  │   Tokens     │    │    (Rust)    │    │    (Rust)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                   │          │
│                              └───────────────────┘          │
│                               underthesea-core              │
└─────────────────────────────────────────────────────────────┘
```

Now both preprocessing and inference are in Rust within the same module, eliminating the C++ dependency entirely.

## Code Changes

The change was minimal from the API perspective:

**Before:**
```python
import pycrfsuite
from underthesea_core import CRFFeaturizer

class FastCRFSequenceTagger:
    def load(self, base_path):
        estimator = pycrfsuite.Tagger()
        estimator.open(model_path)
        # ...
```

**After:**
```python
from underthesea_core import CRFFeaturizer, CRFTagger

class FastCRFSequenceTagger:
    def load(self, base_path):
        estimator = CRFTagger()
        estimator.load(model_path)
        # ...
```

## Benchmark Results

We benchmarked both versions on the same hardware with 100 iterations:

| Function | v9.2.1 (pycrfsuite) | v9.2.5 (Rust) | Improvement |
|----------|---------------------|---------------|-------------|
| word_tokenize | 1.45ms | 1.18ms | **-19%** |
| pos_tag | 3.58ms | 2.93ms | **-18%** |
| ner | 9.61ms | 8.49ms | **-12%** |
| chunk | 6.19ms | 5.65ms | **-9%** |

## Why Is It Faster?

### 1. Unified Runtime

Both `CRFFeaturizer` and `CRFTagger` are now in the same Rust module. This allows:
- Shared memory management
- No intermediate Python object creation
- Potential for future optimizations (e.g., fusing operations)

### 2. Optimized Viterbi Implementation

Our Rust implementation uses pre-allocated vectors and cache-friendly memory layouts:

```rust
fn viterbi(&self, attr_ids: &[Vec<u32>]) -> TaggingResult {
    let n = attr_ids.len();
    let num_labels = self.model.num_labels;

    // Pre-allocated score matrix
    let mut score = vec![vec![f64::NEG_INFINITY; num_labels]; n];
    let mut back = vec![vec![0u32; num_labels]; n];

    // Cache emission scores per position
    let emission_t = self.model.emission_scores(&attr_ids[t]);

    // Direct memory access in inner loop
    for y in 0..num_labels {
        for y_prev in 0..num_labels {
            let trans = self.model.get_transition(y_prev as u32, y as u32);
            // ...
        }
    }
}
```

### 3. Zero-Copy Where Possible

PyO3 bindings allow efficient data transfer between Python and Rust without unnecessary copying.

### 4. Removed Dependency

Removing `python-crfsuite` also means:
- Simpler installation (no C++ compiler needed)
- Smaller package size
- Fewer potential compatibility issues

## Migration Path

The migration was done incrementally across 4 releases:

| Version | Changes |
|---------|---------|
| v9.2.2 | word_tokenize migrated |
| v9.2.3 | pos_tag, ner, chunking migrated |
| v9.2.4 | CRFTrainer migrated, removed unused files |
| v9.2.5 | Removed python-crfsuite dependency |

## Model Compatibility

The Rust implementation can load existing `.crfsuite` model files trained by python-crfsuite. No retraining is required.

```python
# Works with both old and new models
from underthesea import word_tokenize
word_tokenize("Hà Nội là thủ đô của Việt Nam")
# ['Hà Nội', 'là', 'thủ đô', 'của', 'Việt Nam']
```

## Conclusion

By rewriting our CRF inference in Rust and unifying the preprocessing pipeline, we achieved:

- **20% faster inference** across all CRF-based tasks
- **Simpler dependency tree** (no python-crfsuite)
- **Better maintainability** with a single codebase

The full implementation is available in [underthesea-core](https://github.com/undertheseanlp/underthesea/tree/main/extensions/underthesea_core).

## Try It Out

```bash
pip install underthesea==9.2.5
```

```python
from underthesea import word_tokenize, pos_tag, ner, chunk

word_tokenize("Việt Nam")  # 20% faster!
```
