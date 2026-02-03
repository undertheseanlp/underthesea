//! Combined Text Classifier - Keeps entire pipeline in Rust
//!
//! This module provides a unified classifier that performs TF-IDF + SVM
//! without crossing the Python-Rust boundary for intermediate vectors.
//! This eliminates the expensive data transfer overhead.

use crate::svm::{LinearSVC, SparseVec};
use hashbrown::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Optimized TF-IDF Vectorizer for internal use
#[derive(Clone, Serialize, Deserialize)]
struct FastTfIdfVectorizer {
    vocab: HashMap<String, u32>,
    idf: Vec<f32>,
    n_features: usize,
    max_features: usize,
    ngram_range: (usize, usize),
    min_df: usize,
    max_df: f64,
    is_fitted: bool,
}

impl FastTfIdfVectorizer {
    fn new(max_features: usize, ngram_range: (usize, usize), min_df: usize, max_df: f64) -> Self {
        Self {
            vocab: HashMap::with_capacity(max_features),
            idf: Vec::with_capacity(max_features),
            n_features: 0,
            max_features,
            ngram_range,
            min_df,
            max_df,
            is_fitted: false,
        }
    }

    fn fit(&mut self, documents: Vec<String>) {
        let n_docs = documents.len();
        if n_docs == 0 {
            self.is_fitted = true;
            return;
        }

        // Parallel document frequency counting
        let chunk_size = (n_docs / rayon::current_num_threads()).max(100);
        let local_dfs: Vec<HashMap<String, usize>> = documents
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_df: HashMap<String, usize> = HashMap::new();
                let mut seen = HashMap::new();

                for doc in chunk {
                    seen.clear();
                    self.tokenize_into(doc, &mut seen);
                    for term in seen.keys() {
                        *local_df.entry(term.clone()).or_insert(0) += 1;
                    }
                }
                local_df
            })
            .collect();

        // Merge document frequencies
        let mut df: HashMap<String, usize> = HashMap::with_capacity(self.max_features * 2);
        for local in local_dfs {
            for (term, count) in local {
                *df.entry(term).or_insert(0) += count;
            }
        }

        // Filter by min_df and max_df
        let max_df_count = (self.max_df * n_docs as f64) as usize;
        let mut filtered: Vec<(String, usize)> = df
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_count)
            .collect();

        // Sort by frequency (descending) and take top max_features
        filtered.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        filtered.truncate(self.max_features);

        // Build vocabulary and IDF
        self.vocab.clear();
        self.idf.clear();
        self.vocab.reserve(filtered.len());
        self.idf.reserve(filtered.len());

        let n_docs_f = n_docs as f32;
        for (idx, (term, doc_freq)) in filtered.into_iter().enumerate() {
            self.vocab.insert(term, idx as u32);
            let idf_value = ((n_docs_f + 1.0) / (doc_freq as f32 + 1.0)).ln() + 1.0;
            self.idf.push(idf_value);
        }

        self.n_features = self.vocab.len();
        self.is_fitted = true;
    }

    fn vocab_size(&self) -> usize {
        self.n_features
    }

    /// Internal sparse transform (for use within Rust, no FFI overhead)
    #[inline]
    fn transform_sparse_internal(&self, document: &str) -> SparseVec {
        if !self.is_fitted || self.n_features == 0 {
            return Vec::new();
        }

        let mut tf: HashMap<u32, u32> = HashMap::with_capacity(64);
        let mut n_tokens = 0u32;

        self.tokenize_and_count_fast(document, &mut tf, &mut n_tokens);

        if n_tokens == 0 {
            return Vec::new();
        }

        let n_tokens_f = n_tokens as f32;
        let mut result: SparseVec = Vec::with_capacity(tf.len());
        let mut norm_sq = 0.0f32;

        for (&idx, &count) in &tf {
            let tf_value = count as f32 / n_tokens_f;
            let tfidf = tf_value * self.idf[idx as usize];
            norm_sq += tfidf * tfidf;
            result.push((idx, tfidf));
        }

        // L2 normalize
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for (_, v) in &mut result {
                *v *= inv_norm;
            }
        }

        result
    }

    /// Fast tokenize and count using byte offsets
    #[inline]
    fn tokenize_and_count_fast(
        &self,
        document: &str,
        tf: &mut HashMap<u32, u32>,
        n_tokens: &mut u32,
    ) {
        let bytes = document.as_bytes();
        let mut word_starts: Vec<usize> = Vec::with_capacity(32);
        let mut word_ends: Vec<usize> = Vec::with_capacity(32);

        let mut in_word = false;
        let mut start = 0;

        for (i, &b) in bytes.iter().enumerate() {
            let is_space = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
            if !is_space && !in_word {
                start = i;
                in_word = true;
            } else if is_space && in_word {
                word_starts.push(start);
                word_ends.push(i);
                in_word = false;
            }
        }
        if in_word {
            word_starts.push(start);
            word_ends.push(bytes.len());
        }

        let n_words = word_starts.len();
        if n_words == 0 {
            return;
        }

        // Unigrams
        if self.ngram_range.0 <= 1 && self.ngram_range.1 >= 1 {
            for i in 0..n_words {
                let word = &document[word_starts[i]..word_ends[i]];
                if let Some(&idx) = self.vocab.get(word) {
                    *tf.entry(idx).or_insert(0) += 1;
                    *n_tokens += 1;
                }
            }
        }

        // Bigrams
        if self.ngram_range.0 <= 2 && self.ngram_range.1 >= 2 && n_words >= 2 {
            let mut bigram_buf = String::with_capacity(64);
            for i in 0..n_words - 1 {
                bigram_buf.clear();
                bigram_buf.push_str(&document[word_starts[i]..word_ends[i]]);
                bigram_buf.push(' ');
                bigram_buf.push_str(&document[word_starts[i + 1]..word_ends[i + 1]]);

                if let Some(&idx) = self.vocab.get(&bigram_buf) {
                    *tf.entry(idx).or_insert(0) += 1;
                    *n_tokens += 1;
                }
            }
        }

        // Higher n-grams
        for n in 3..=self.ngram_range.1 {
            if n > n_words || n < self.ngram_range.0 {
                continue;
            }
            let mut ngram_buf = String::with_capacity(n * 16);
            for i in 0..=n_words - n {
                ngram_buf.clear();
                for j in 0..n {
                    if j > 0 {
                        ngram_buf.push(' ');
                    }
                    ngram_buf.push_str(&document[word_starts[i + j]..word_ends[i + j]]);
                }
                if let Some(&idx) = self.vocab.get(&ngram_buf) {
                    *tf.entry(idx).or_insert(0) += 1;
                    *n_tokens += 1;
                }
            }
        }
    }

    /// Tokenize document into a set of unique tokens (for fit())
    #[inline]
    fn tokenize_into(&self, document: &str, seen: &mut HashMap<String, ()>) {
        let words: Vec<&str> = document.split_whitespace().collect();
        let n_words = words.len();

        if n_words == 0 {
            return;
        }

        // Unigrams
        if self.ngram_range.0 <= 1 && self.ngram_range.1 >= 1 {
            for &word in &words {
                seen.entry(word.to_string()).or_insert(());
            }
        }

        // Bigrams
        if self.ngram_range.0 <= 2 && self.ngram_range.1 >= 2 && n_words >= 2 {
            let mut bigram_buf = String::with_capacity(64);
            for i in 0..n_words - 1 {
                bigram_buf.clear();
                bigram_buf.push_str(words[i]);
                bigram_buf.push(' ');
                bigram_buf.push_str(words[i + 1]);
                seen.entry(bigram_buf.clone()).or_insert(());
            }
        }

        // Higher n-grams
        for n in 3..=self.ngram_range.1 {
            if n > n_words || n < self.ngram_range.0 {
                continue;
            }
            let mut ngram_buf = String::with_capacity(n * 16);
            for i in 0..=n_words - n {
                ngram_buf.clear();
                for j in 0..n {
                    if j > 0 {
                        ngram_buf.push(' ');
                    }
                    ngram_buf.push_str(words[i + j]);
                }
                seen.entry(ngram_buf.clone()).or_insert(());
            }
        }
    }
}

/// Fast Text Classifier combining TF-IDF and Linear SVM
///
/// Keeps the entire pipeline in Rust to avoid FFI overhead.
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct TextClassifier {
    vectorizer: FastTfIdfVectorizer,
    model: Option<LinearSVC>,
    c: f32,
    max_iter: usize,
    tol: f32,
}

impl Default for TextClassifier {
    fn default() -> Self {
        Self::new(20000, (1, 2), 1, 1.0, 1.0, 1000, 0.1)
    }
}

#[pymethods]
impl TextClassifier {
    #[new]
    #[pyo3(signature = (max_features=20000, ngram_range=(1, 2), min_df=1, max_df=1.0, c=1.0, max_iter=1000, tol=0.1))]
    pub fn new(
        max_features: usize,
        ngram_range: (usize, usize),
        min_df: usize,
        max_df: f64,
        c: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            vectorizer: FastTfIdfVectorizer::new(max_features, ngram_range, min_df, max_df),
            model: None,
            c: c as f32,
            max_iter,
            tol: tol as f32,
        }
    }

    /// Fit the classifier on training data
    pub fn fit(&mut self, texts: Vec<String>, labels: Vec<String>) {
        let n_samples = texts.len();
        if n_samples == 0 {
            return;
        }

        // Step 1: Fit vectorizer
        self.vectorizer.fit(texts.clone());

        let n_features = self.vectorizer.vocab_size();
        if n_features == 0 {
            return;
        }

        // Step 2: Transform to sparse vectors
        let sparse_features: Vec<SparseVec> = texts
            .par_iter()
            .map(|text| self.vectorizer.transform_sparse_internal(text))
            .collect();

        // Precompute ||x_i||^2
        let x_sq_norms: Vec<f32> = sparse_features
            .par_iter()
            .map(|x| x.iter().map(|&(_, v)| v * v).sum())
            .collect();

        // Step 3: Get unique classes
        let mut classes: Vec<String> = labels.to_vec();
        classes.sort();
        classes.dedup();
        let n_classes = classes.len();

        let class_to_idx: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        let y_idx: Vec<usize> = labels.iter().map(|l| class_to_idx[l]).collect();

        // Step 4: Train binary classifiers in parallel
        let results: Vec<(Vec<f32>, f32)> = (0..n_classes)
            .into_par_iter()
            .map(|class_idx| {
                let y_binary: Vec<i8> = y_idx
                    .iter()
                    .map(|&idx| if idx == class_idx { 1 } else { -1 })
                    .collect();

                solve_l2r_l2_svc(
                    &sparse_features,
                    &y_binary,
                    &x_sq_norms,
                    n_features,
                    self.c,
                    self.tol,
                    self.max_iter,
                )
            })
            .collect();

        let weights: Vec<Vec<f32>> = results.iter().map(|(w, _)| w.clone()).collect();
        let biases: Vec<f32> = results.iter().map(|(_, b)| *b).collect();

        self.model = Some(LinearSVC::from_weights(
            weights, biases, classes, n_features,
        ));
    }

    /// Predict labels for multiple texts (parallel)
    pub fn predict_batch(&self, texts: Vec<String>) -> Vec<String> {
        let model = match &self.model {
            Some(m) => m,
            None => return vec!["".to_string(); texts.len()],
        };

        let classes = model.classes();

        let indices: Vec<usize> = texts
            .par_iter()
            .map(|text| {
                let sparse = self.vectorizer.transform_sparse_internal(text);
                model.predict_sparse_idx(&sparse)
            })
            .collect();

        indices
            .into_iter()
            .map(|idx| classes[idx].clone())
            .collect()
    }

    /// Predict label for a single text
    pub fn predict(&self, text: &str) -> String {
        let model = match &self.model {
            Some(m) => m,
            None => return "".to_string(),
        };

        let sparse = self.vectorizer.transform_sparse_internal(text);
        model.predict_sparse_internal(&sparse)
    }

    /// Predict with confidence scores
    pub fn predict_with_scores(&self, texts: Vec<String>) -> Vec<(String, f64)> {
        let model = match &self.model {
            Some(m) => m,
            None => return texts.iter().map(|_| ("".to_string(), 0.0)).collect(),
        };

        texts
            .par_iter()
            .map(|text| {
                let sparse = self.vectorizer.transform_sparse_internal(text);
                model.predict_sparse_with_score_internal(&sparse)
            })
            .collect()
    }

    #[getter]
    pub fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.vectorizer.vocab_size()
    }

    #[getter]
    pub fn classes(&self) -> Vec<String> {
        self.model.as_ref().map(|m| m.classes()).unwrap_or_default()
    }

    /// Save model to binary file
    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// Load model from binary file
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let reader = BufReader::new(file);
        let clf: Self = bincode::deserialize_from(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(clf)
    }
}

/// LIBLINEAR's solve_l2r_l2_svc - Dual Coordinate Descent
#[inline(never)]
fn solve_l2r_l2_svc(
    x: &[SparseVec],
    y: &[i8],
    x_sq_norms: &[f32],
    n_features: usize,
    c: f32,
    eps: f32,
    max_iter: usize,
) -> (Vec<f32>, f32) {
    let n = x.len();
    let diag = 0.5 / c;
    let qd: Vec<f32> = x_sq_norms.iter().map(|&xn| xn + diag).collect();

    let mut alpha = vec![0.0f32; n];
    let mut w = vec![0.0f32; n_features];
    let mut index: Vec<usize> = (0..n).collect();

    for iter in 0..max_iter {
        // Shuffle indices
        for i in 0..n {
            let j = i + (iter * 1103515245 + 12345) % (n - i).max(1);
            index.swap(i, j);
        }

        let mut max_violation = 0.0f32;

        for &i in &index {
            let yi = y[i] as f32;
            let xi = &x[i];

            let wxi: f32 = xi.iter().map(|&(j, v)| w[j as usize] * v).sum();
            let g = yi * wxi - 1.0 + diag * alpha[i];
            let pg = if alpha[i] == 0.0 { g.min(0.0) } else { g };

            max_violation = max_violation.max(pg.abs());

            if pg.abs() > 1e-12 {
                let alpha_old = alpha[i];
                alpha[i] = (alpha[i] - g / qd[i]).max(0.0);

                let d = (alpha[i] - alpha_old) * yi;
                if d.abs() > 1e-12 {
                    for &(j, v) in xi.iter() {
                        w[j as usize] += d * v;
                    }
                }
            }
        }

        if max_violation <= eps {
            break;
        }
    }

    // Compute bias
    let mut bias_sum = 0.0f32;
    let mut n_sv = 0;

    for i in 0..n {
        if alpha[i] > 1e-8 {
            let yi = y[i] as f32;
            let wxi: f32 = x[i].iter().map(|&(j, v)| w[j as usize] * v).sum();
            bias_sum += yi * (1.0 - alpha[i] * diag) - wxi;
            n_sv += 1;
        }
    }

    let bias = if n_sv > 0 {
        bias_sum / n_sv as f32
    } else {
        0.0
    };

    (w, bias)
}
