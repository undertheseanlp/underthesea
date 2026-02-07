---
slug: rewrite-rust-crf-model
title: Rewriting CRF Model in Rust
authors: [rain1024]
tags: [rust, performance, crf, nlp]
---

# Rewriting CRF Model in Rust: A 20% Performance Boost

In underthesea v9.2.5, we completed the migration from `python-crfsuite` to our native Rust implementation `underthesea-core`. This change resulted in a **20% performance improvement** across all CRF-based NLP tasks, plus up to **10x faster training** through systematic optimization.

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

## Inference Benchmark Results

We benchmarked both versions on the same hardware with 100 iterations:

| Function | v9.2.1 (pycrfsuite) | v9.2.5 (Rust) | Improvement |
|----------|---------------------|---------------|-------------|
| word_tokenize | 1.45ms | 1.18ms | **-19%** |
| pos_tag | 3.58ms | 2.93ms | **-18%** |
| ner | 9.61ms | 8.49ms | **-12%** |
| chunk | 6.19ms | 5.65ms | **-9%** |

## Why Is Inference Faster?

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

## Training Optimizations

Beyond inference, we also optimized the CRF trainer in `underthesea-core`. The original Rust trainer was **7.2x slower** than python-crfsuite for word segmentation. Through four key optimizations, we made it competitive — and even faster for some tasks.

### CRF Training Algorithm

The trainer uses L-BFGS optimization with OWL-QN extension for L1 regularization. The core computation is the forward-backward algorithm with complexity **O(n × L²)** per sequence, where n is the sequence length and L is the number of labels.

Following CRFsuite's approach, we use **scaled probability space** instead of log-space — avoiding log/exp in inner loops while maintaining numerical stability with scaling factors.

### Starting Point

| Task | python-crfsuite | underthesea-core (original) | Slowdown |
|------|-----------------|----------------------------|----------|
| Word Segmentation | 2m 34s | 18m 33s | **7.2x slower** |
| POS Tagging | 4m 50s | 7m 21s | **1.5x slower** |

### Optimization 1: Flat Data Structure for Feature Lookup

The original used nested vectors (`Vec<Vec<(u32, u32)>>`) for feature lookup — each inner `Vec` separately heap-allocated, causing cache misses for large feature sets (562k features).

We flattened into contiguous arrays with offset indexing:

```rust
// Contiguous memory, excellent cache locality
attr_offsets: Vec<u32>               // attr_id -> start index
attr_features_flat: Vec<(u32, u32)>  // flattened (label_id, feature_id) pairs
```

**Result**: Word segmentation went from 18m 33s to 1m 49s — a **10.2x speedup**.

### Optimization 2: Loop Unrolling for Auto-Vectorization

The forward-backward algorithm has O(n × L²) inner loops. For POS tagging (16 labels = 256 transitions per timestep), we applied 4-way manual loop unrolling to enable SIMD auto-vectorization.

**Result**: POS tagging (10 iterations) went from 25.7s to 17.58s — a **1.46x speedup**.

### Optimization 3: Unsafe Bounds-Check Elimination

We used `unsafe` with `get_unchecked` for hot paths where indices are provably valid, eliminating Rust's bounds checks in tight loops. All `unsafe` blocks are guarded by loop bounds derived from array lengths and algorithm invariants.

### Optimization 4: Fused Operations

Separate loops for related operations (scale, sum, normalize) were fused into single passes to reduce redundant memory traversals.

### Cumulative Optimization Impact

| Optimization | Word Seg Speedup | POS Tag Speedup |
|--------------|------------------|-----------------|
| Baseline (original) | 1.0x | 1.0x |
| + Flat data structure | **10.2x** | 1.4x |
| + Loop unrolling | 10.2x | **2.1x** |
| + Unsafe bounds elim | ~10.2x | ~2.3x |
| **Total** | **10.2x** | **2.3x** |

### Training Benchmark Results (200 iterations)

| Task | Features | Labels | python-crfsuite | underthesea-core | Result |
|------|----------|--------|-----------------|------------------|--------|
| Word Segmentation | 562,885 | 2 | 2m 2s | **1m 38s** | **1.24x faster** |
| POS Tagging | 626,723 | 16 | 4m 3s | 4m 14s | ~equal (4% slower) |

### Accuracy Verification

| Task | Metric | python-crfsuite | underthesea-core |
|------|--------|-----------------|------------------|
| Word Segmentation | Syllable Accuracy | 98.89% | **98.89%** |
| Word Segmentation | Word F1 | 98.00% | **98.00%** |
| POS Tagging | Accuracy | 95.98% | **95.97%** |

**Accuracy is identical** — optimizations only affected performance, not correctness.

## What Didn't Work

We also evaluated several optimizations that did not provide significant improvements:

- **Explicit SIMD Intrinsics (AVX2)**: The inner loops process only 2-16 labels, too small for explicit SIMD to outperform the compiler's auto-vectorization with loop unrolling.
- **Parallel Forward-Backward (Rayon)**: Thread-local gradient accumulation overhead from buffer allocation per sequence and gradient merging negated the parallelism benefits.
- **Compressed Sparse Features**: The flat data structure with offset indexing already provides efficient sparse feature access — additional compression just adds decode overhead.

## Key Insight: Different Tasks, Different Bottlenecks

| Feature Set Size | Bottleneck | Best Optimization |
|------------------|------------|-------------------|
| Large (500k+) | Feature lookup (cache misses) | Flat data structure |
| Small (&lt;50k) | Forward-backward O(L²) | Loop unrolling |

### Why python-crfsuite Was Initially Faster

CRFsuite (C implementation) already had hand-optimized sparse feature storage, SIMD-vectorized matrix operations (AVX/SSE intrinsics), cache-optimized memory layouts, and decades of optimization. Our flat data structure and loop unrolling effectively replicated these advantages in Rust.

### Lessons Learned

1. **Profile first** — the bottleneck was different for each task
2. **Data structure matters** — flat arrays beat nested vectors by 10x
3. **Cache locality is critical** — sequential memory access enables hardware prefetching
4. **Unsafe Rust is justified** — when correctness is provable and performance is critical
5. **Incremental migration reduces risk** — migrating one task at a time allowed validation at each step

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

By rewriting our CRF implementation in Rust and unifying the pipeline, we achieved:

- **12-19% faster inference** across all CRF-based tasks
- **1.24x faster training** for word segmentation (10x from original Rust implementation)
- **Identical accuracy** — no degradation from the migration
- **Simpler dependency tree** (no python-crfsuite / C++ compiler needed)
- **Better maintainability** with a single Rust codebase

The full implementation is available in [underthesea-core](https://github.com/undertheseanlp/underthesea/tree/main/extensions/underthesea_core).

## Try It Out

```bash
pip install underthesea==9.2.5
```

```python
from underthesea import word_tokenize, pos_tag, ner, chunk

word_tokenize("Việt Nam")  # 20% faster!
```
