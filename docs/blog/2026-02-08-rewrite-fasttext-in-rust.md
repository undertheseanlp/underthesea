---
slug: rewrite-fasttext-in-rust
title: Rewriting FastText in Rust
authors: [rain1024]
tags: [rust, performance, fasttext, nlp, language-detection]
---

# Rewriting FastText in Rust: 1,149 Lines to Replace a C++ Dependency

In underthesea v9.2.9, we replaced the `fasttext` Python package (a wrapper around Facebook's C++ library) with a **pure Rust implementation** inside `underthesea-core`. The result: identical predictions, simpler installation, and one fewer C++ dependency in our stack.

<!-- truncate -->

## Why Replace FastText?

FastText is used in underthesea for **language detection** — identifying whether input text is Vietnamese, English, French, etc. The original setup worked, but had friction:

1. **C++ compilation required** — `pip install fasttext` needs a C++ compiler, which fails on minimal Docker images and Windows environments
2. **Heavy dependency** — pulls in the entire FastText C++ library (~50MB) just for inference
3. **Multiple language boundaries** — Python → C++ FFI → Python, with overhead at each crossing
4. **Maintenance burden** — the `fasttext` package has had compatibility issues with newer Python versions

Since we already had `underthesea-core` (our Rust extension via PyO3), adding FastText inference there was a natural fit.

## What We Built

A **pure Rust FastText inference engine** in 1,149 lines across 6 files:

```
extensions/underthesea_core/src/fasttext/
├── mod.rs          # FastTextModel: load + predict (126 lines)
├── args.rs         # Model hyperparameters deserialization (127 lines)
├── dictionary.rs   # Vocabulary + n-gram hashing (285 lines)
├── inference.rs    # Hierarchical softmax + softmax prediction (303 lines)
├── matrix.rs       # Dense + quantized matrix operations (256 lines)
└── hash.rs         # FNV-1a hash (52 lines)
```

This is inference-only — we load existing FastText `.bin` and `.ftz` models trained by the original C++ library. No retraining needed.

## The Architecture Change

### Before

```
Python
  │
  ├── import fasttext           # C++ FFI wrapper
  │     └── libfasttext.so      # Facebook's C++ library
  │           └── model.bin     # FastText binary model
  │
  └── predictions
```

Three language boundaries: Python → C++ wrapper → C++ library.

### After

```
Python
  │
  ├── from underthesea_core import FastText   # PyO3 Rust binding
  │     └── fasttext::FastTextModel           # Pure Rust
  │           └── model.bin                   # Same binary format
  │
  └── predictions
```

One clean boundary: Python → Rust via PyO3.

### Code Change

The API is almost identical:

**Before:**
```python
import fasttext

model = fasttext.load_model("lid.176.ftz")
predictions = model.predict("Xin chào thế giới")
# (('__label__vi',), array([0.98]))
```

**After:**
```python
from underthesea_core import FastText

model = FastText.load("lid.176.ftz")
predictions = model.predict("Xin chào thế giới")
# [('vi', 0.98)]
```

In underthesea's `lang_detect` pipeline, the switch was a one-line import change:

```python
# Before: import fasttext
# After:
from underthesea_core import FastText

lang_detect_model = FastText.load(str(model_path))
predictions = lang_detect_model.predict(text, k=1)
```

## Implementation Deep Dive

### 1. Binary Format Parsing

FastText models use a custom binary format. We must parse it byte-for-byte to match the C++ implementation:

```rust
pub fn load(path: &str) -> io::Result<Self> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let quant = path.ends_with(".ftz");

    check_header(&mut reader)?;          // Magic number + version
    let args = Args::load(&mut reader)?;  // Hyperparameters
    let dictionary = Dictionary::load(&mut reader, &args)?;  // Vocabulary

    let file_quant = read_bool(&mut reader)?;
    let input_matrix = FastTextMatrix::load(&mut reader, quant && file_quant)?;

    let output_quant = read_bool(&mut reader)?;
    let output_matrix = FastTextMatrix::load(&mut reader, quant && output_quant)?;

    // Build Huffman tree for hierarchical softmax
    let hs_tree = if args.loss == LossName::HierarchicalSoftmax {
        let counts = dictionary.get_label_counts();
        Some(HSTree::build(&counts))
    } else {
        None
    };

    Ok(FastTextModel { args, dictionary, input_matrix, output_matrix, hs_tree })
}
```

The magic number `0x2F4F16BA` and version `12` must match exactly, or the model is rejected.

### 2. The Hash Function: A Subtle Trap

FastText uses FNV-1a hashing for word and n-gram lookups. The critical detail: **non-ASCII bytes are sign-extended**, matching C++'s signed `char` behavior:

```rust
pub fn fasttext_hash(s: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &byte in s {
        // Sign-extend: cast to i8 first (like C++ int8_t), then to u32
        h ^= byte as i8 as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}
```

Without the `byte as i8 as u32` cast, Vietnamese characters (which use multi-byte UTF-8 with high bytes like `0xC3`, `0xE1`) would hash to different values than the C++ implementation, producing wrong predictions. This was the trickiest part to get right.

### 3. Dictionary and N-gram Features

The dictionary handles vocabulary lookup with an open-addressing hash table (30M slots, matching C++):

```rust
pub struct Dictionary {
    entries: Vec<Entry>,
    word2int: Vec<i32>,    // 30M-slot hash table
    nwords: i32,
    nlabels: i32,
    pruneidx: HashMap<i32, i32>,  // For quantized models
    // N-gram parameters
    bucket: i32,
    minn: i32,
    maxn: i32,
    word_ngrams: i32,
}
```

For each input token, we generate three types of features:
1. **Word ID** — direct vocabulary lookup
2. **Character n-grams** — subword features like `<xin`, `xin>`, `<xi`, `in>` for the word "xin"
3. **Word n-grams** — bigrams/trigrams of word IDs

Character n-grams are bounded by `<` and `>` markers and hashed into buckets:

```rust
fn compute_char_ngrams(&self, word: &str, features: &mut Vec<i32>) {
    let bounded = format!("<{}>", word);
    let bytes = bounded.as_bytes();

    // Walk character boundaries (not bytes — Vietnamese is multi-byte UTF-8)
    let char_boundaries = compute_utf8_boundaries(bytes);

    for n in self.minn..=self.maxn {
        for start_char in 0..=(nchars - n) {
            let ngram = &bytes[char_boundaries[start_char]..char_boundaries[start_char + n]];
            let h = fasttext_hash(ngram);
            let bucket_hash = (h as i64 % self.bucket as i64) as i32;
            self.push_hash(features, bucket_hash);
        }
    }
}
```

This is where FastText's power comes from — even unknown words get meaningful features from their character substrings.

### 4. Two Inference Paths

FastText models use different loss functions. We implement both:

**Hierarchical Softmax** (for models with many labels like language detection with 176 languages):

The key insight: instead of computing scores for all 176 labels (O(n)), we traverse a Huffman tree via DFS, pruning branches that can't beat the current top-k (O(k log n)):

```rust
fn dfs(&self, k: usize, threshold: f32, node: usize, score: f32,
       hidden: &[f32], output: &FastTextMatrix,
       heap: &mut BinaryHeap<Reverse<(FloatOrd, usize)>>) {
    // Prune: if this branch can't beat threshold, skip
    if score < std_log(threshold) { return; }
    // Prune: if heap is full and this can't beat worst result, skip
    if heap.len() == k && score < heap.peek().unwrap().0.0.0 { return; }

    let n = &self.tree[node];
    if n.left == -1 && n.right == -1 {
        // Leaf node = label
        heap.push(Reverse((FloatOrd(score), node)));
        if heap.len() > k { heap.pop(); }
        return;
    }

    let f = sigmoid(output.dot_row(hidden, node - self.osz));

    // Recurse into both children with accumulated log-probabilities
    self.dfs(k, threshold, n.left as usize,  score + std_log(1.0 - f), hidden, output, heap);
    self.dfs(k, threshold, n.right as usize, score + std_log(f),       hidden, output, heap);
}
```

**Standard Softmax** (for models with fewer labels):

Direct computation — one dot product per label, then partial sort to find top-k:

```rust
fn predict_softmax(k: usize, hidden: &[f32], output: &FastTextMatrix, nlabels: usize)
    -> Vec<(f32, usize)>
{
    let mut logits: Vec<f32> = (0..nlabels)
        .map(|i| output.dot_row(hidden, i))
        .collect();
    softmax(&mut logits);

    // Partial sort: O(n) instead of O(n log n) full sort
    let mut indices: Vec<usize> = (0..nlabels).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b|
        logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Equal)
    );
    indices.truncate(k);
    // ...
}
```

### 5. Dense and Quantized Matrices

FastText models come in two flavors:
- **`.bin`** — dense float32 matrices (large but fast)
- **`.ftz`** — product-quantized matrices (4-10x smaller, slightly slower)

We handle both through a trait:

```rust
pub trait Matrix {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn add_row_to(&self, row: usize, output: &mut [f32]);
    fn dot_row(&self, vec: &[f32], row: usize) -> f32;
}
```

Dense is straightforward — row-major float array with SIMD-friendly dot products:

```rust
fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
    let start = row * self.cols;
    let row_data = &self.data[start..start + self.cols];
    vec.iter().zip(row_data.iter()).map(|(&a, &b)| a * b).sum()
}
```

Quantized uses product quantization — each vector is split into sub-spaces, each quantized to one of 256 centroids (8-bit codes):

```rust
fn dot_row(&self, vec: &[f32], row: usize) -> f32 {
    let norm = self.get_norm(row);
    let mut sum = 0.0f32;
    let mut dim_offset = 0;
    for subq in 0..self.pq.nsubq {
        let code = self.codes[row * self.pq.nsubq + subq];
        let centroid = self.pq.get_centroid(subq, code);
        for (i, &c) in centroid.iter().enumerate() {
            sum += vec[dim_offset + i] * c;
        }
        dim_offset += self.pq.dsub;
    }
    sum * norm
}
```

This lets us load Facebook's compressed `lid.176.ftz` (917KB) instead of the dense `lid.176.bin` (~130MB).

### 6. PyO3 Bindings

The Python interface is minimal — PyO3 makes it trivial:

```rust
#[pyclass(name = "FastText")]
pub struct PyFastText {
    model: fasttext::FastTextModel,
}

#[pymethods]
impl PyFastText {
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let model = fasttext::FastTextModel::load(path)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(Self { model })
    }

    #[pyo3(signature = (text, k=1))]
    pub fn predict(&self, text: &str, k: usize) -> Vec<(String, f32)> {
        self.model.predict(text, k)
    }

    pub fn get_labels(&self) -> Vec<String> {
        self.model.get_labels()
    }

    #[getter]
    pub fn dim(&self) -> i32 { self.model.dim() }

    #[getter]
    pub fn nwords(&self) -> i32 { self.model.nwords() }

    #[getter]
    pub fn nlabels(&self) -> i32 { self.model.nlabels() }
}
```

## What We Gained

### Simpler Installation

```
# Before: needs C++ compiler, can fail on many platforms
pip install fasttext  # Often fails: "error: command 'gcc' failed"

# After: pre-built wheels, just works
pip install underthesea  # Includes Rust-compiled underthesea-core
```

### Fewer Dependencies

```
# Before: underthesea required
fasttext>=0.9.2      # C++ FastText wrapper
python-crfsuite      # C++ CRFsuite wrapper
scikit-learn         # For TfidfVectorizer + LinearSVC
joblib               # For pickle serialization

# After: just one Rust extension
underthesea-core     # Everything in Rust
```

All four dependencies — `fasttext`, `python-crfsuite`, `scikit-learn` (for classification), and `joblib` — have been replaced by a single `underthesea-core` package.

### Model Compatibility

The Rust implementation reads the **exact same binary format** as Facebook's C++ code. Any `.bin` or `.ftz` model trained by the original FastText works unchanged:

```python
from underthesea_core import FastText

# Facebook's pre-trained language identification model — just works
model = FastText.load("lid.176.ftz")
model.predict("Hà Nội là thủ đô của Việt Nam", k=3)
# [('vi', 0.97), ('id', 0.01), ('ms', 0.005)]
```

## The Bigger Picture: Replacing C++ with Rust

This FastText rewrite is part of a broader effort to consolidate underthesea's backend into a single Rust extension. Here's the full migration timeline:

| Version | What Changed | C++ Dependency Removed |
|---------|-------------|----------------------|
| v9.2.2-v9.2.5 | CRF tagger rewritten in Rust | `python-crfsuite` |
| v9.2.9 | Text classifier rewritten in Rust | `scikit-learn` + `joblib` |
| v9.2.9 | FastText inference rewritten in Rust | `fasttext` |

### Lines of Rust replacing each dependency:

| Component | Rust Lines | Replaces |
|-----------|-----------|----------|
| FastText inference | 1,149 | `fasttext` (C++ FFI) |
| TF-IDF + Linear SVM | 2,024 | `scikit-learn` + `joblib` |
| CRF tagger + trainer | ~3,000 | `python-crfsuite` (C++ FFI) |
| Vietnamese preprocessor | 620 | Custom Python code |
| PyO3 bindings | 918 | N/A |
| **Total** | **~7,700** | **4 C/C++ dependencies** |

Under 8,000 lines of Rust replaced four separate C/C++ dependencies, unifying everything into a single compiled extension with:
- Pre-built wheels for Linux, macOS (Intel + ARM), and Windows
- No C/C++ compiler needed for installation
- One coherent codebase instead of four upstream projects
- Consistent binary serialization (bincode) instead of mixed pickle/joblib/custom formats

## Lessons Learned

### 1. Byte-level compatibility is non-negotiable

The hash function sign-extension bug (`byte as i8 as u32`) took the longest to find. Without it, predictions for any non-ASCII text (i.e., all Vietnamese) were wrong. When reimplementing binary formats, every byte matters.

### 2. Inference-only is the sweet spot

We deliberately chose not to implement FastText training in Rust. Training happens once; inference happens millions of times. By supporting the existing `.bin`/`.ftz` format, we get the benefits of Rust for the hot path while still using Facebook's battle-tested training code when needed.

### 3. Product quantization support is essential

Many production FastText models use `.ftz` (quantized) format for 4-10x size reduction. Skipping quantization support would have made the rewrite impractical for real deployments.

### 4. PyO3 makes Rust-Python integration painless

The binding layer is under 50 lines. PyO3 handles type conversion, error propagation, GIL management, and memory cleanup automatically. The cognitive overhead of maintaining a Rust extension is surprisingly low.

## Try It Out

```bash
pip install underthesea>=9.2.9
```

```python
from underthesea import lang_detect

# Uses Rust FastText under the hood
lang_detect("Xin chào thế giới")  # 'vi'
lang_detect("Hello world")        # 'en'
lang_detect("Bonjour le monde")   # 'fr'
```

Or use the Rust FastText directly:

```python
from underthesea_core import FastText

model = FastText.load("your_model.bin")
model.predict("your text here", k=5)
model.get_labels()   # All available labels
model.dim            # Embedding dimension
model.nwords         # Vocabulary size
model.nlabels        # Number of labels
```

## Links

- [PR #953](https://github.com/undertheseanlp/underthesea/pull/953) — Remove fasttext dependency
- [PR #947](https://github.com/undertheseanlp/underthesea/pull/947) — Add pure Rust FastText inference
- [underthesea-core source](https://github.com/undertheseanlp/underthesea/tree/main/extensions/underthesea_core) — Full Rust implementation
- [underthesea-core on PyPI](https://pypi.org/project/underthesea-core/) — Pre-built wheels
