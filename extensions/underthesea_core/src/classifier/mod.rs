//! Combined Text Classifier - Keeps entire pipeline in Rust
//!
//! This module provides a unified classifier that performs TF-IDF + SVM
//! without crossing the Python-Rust boundary for intermediate vectors.
//! This eliminates the expensive data transfer overhead.

use crate::preprocessor::TextPreprocessor;
use crate::svm::{LinearSVC, SparseVec};
use hashbrown::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Label class compatible with underthesea API
#[pyclass]
#[derive(Clone)]
pub struct Label {
    #[pyo3(get, set)]
    pub value: String,
    #[pyo3(get, set)]
    pub score: f64,
}

#[pymethods]
impl Label {
    #[new]
    #[pyo3(signature = (value, score=1.0))]
    pub fn new(value: String, score: f64) -> Self {
        let score = score.clamp(0.0, 1.0);
        Self { value, score }
    }

    fn __str__(&self) -> String {
        format!("{} ({:.4})", self.value, self.score)
    }

    fn __repr__(&self) -> String {
        format!("{} ({:.4})", self.value, self.score)
    }
}

/// Sentence class compatible with underthesea API
#[pyclass]
#[derive(Clone)]
pub struct Sentence {
    #[pyo3(get, set)]
    pub text: String,
    #[pyo3(get, set)]
    pub labels: Vec<Label>,
}

#[pymethods]
impl Sentence {
    #[new]
    #[pyo3(signature = (text=String::new(), labels=None))]
    pub fn new(text: String, labels: Option<Vec<Label>>) -> Self {
        Self {
            text,
            labels: labels.unwrap_or_default(),
        }
    }

    fn __str__(&self) -> String {
        let labels_str: Vec<String> = self.labels.iter().map(|l| l.__str__()).collect();
        format!(
            "Sentence: \"{}\" - Labels: [{}]",
            self.text,
            labels_str.join(", ")
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    /// Add labels to the sentence
    pub fn add_label(&mut self, label: Label) {
        self.labels.push(label);
    }

    /// Add multiple labels to the sentence
    pub fn add_labels(&mut self, labels: Vec<Label>) {
        self.labels.extend(labels);
    }
}

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
    /// Optional preprocessor — when set, fit/predict auto-preprocess texts.
    /// Serialized with bincode alongside the model.
    preprocessor: Option<TextPreprocessor>,
}

impl Default for TextClassifier {
    fn default() -> Self {
        Self::new(20000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, None)
    }
}

#[pymethods]
impl TextClassifier {
    #[new]
    #[pyo3(signature = (max_features=20000, ngram_range=(1, 2), min_df=1, max_df=1.0, c=1.0, max_iter=1000, tol=0.1, preprocessor=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_features: usize,
        ngram_range: (usize, usize),
        min_df: usize,
        max_df: f64,
        c: f64,
        max_iter: usize,
        tol: f64,
        preprocessor: Option<TextPreprocessor>,
    ) -> Self {
        Self {
            vectorizer: FastTfIdfVectorizer::new(max_features, ngram_range, min_df, max_df),
            model: None,
            c: c as f32,
            max_iter,
            tol: tol as f32,
            preprocessor,
        }
    }

    /// Fit the classifier on training data.
    /// If preprocessor is set, texts are automatically preprocessed.
    pub fn fit(&mut self, texts: Vec<String>, labels: Vec<String>) {
        let n_samples = texts.len();
        if n_samples == 0 {
            return;
        }

        // Auto-preprocess if preprocessor is configured
        let texts = self.preprocess_texts(texts);

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

    /// Predict labels for multiple texts (parallel).
    /// If preprocessor is set, texts are automatically preprocessed.
    pub fn predict_batch(&self, texts: Vec<String>) -> Vec<String> {
        let model = match &self.model {
            Some(m) => m,
            None => return vec!["".to_string(); texts.len()],
        };

        let texts = self.preprocess_texts(texts);
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

    /// Predict label for a single text.
    /// If preprocessor is set, text is automatically preprocessed.
    pub fn predict(&self, text: &str) -> String {
        let model = match &self.model {
            Some(m) => m,
            None => return "".to_string(),
        };

        let text = self.preprocess_text(text);
        let sparse = self.vectorizer.transform_sparse_internal(&text);
        model.predict_sparse_internal(&sparse)
    }

    /// Predict label with confidence score.
    /// If preprocessor is set, text is automatically preprocessed.
    pub fn predict_with_score(&self, text: &str) -> (String, f64) {
        let model = match &self.model {
            Some(m) => m,
            None => return ("".to_string(), 0.0),
        };

        let text = self.preprocess_text(text);
        let sparse = self.vectorizer.transform_sparse_internal(&text);
        model.predict_sparse_with_score_internal(&sparse)
    }

    /// Predict and add labels to a Sentence object (underthesea-compatible API)
    pub fn predict_sentence(&self, sentence: &mut Sentence) {
        let (label_value, score) = self.predict_with_score(&sentence.text);
        sentence.labels.clear();
        sentence.labels.push(Label::new(label_value, score));
    }

    /// Predict with confidence scores (batch).
    /// If preprocessor is set, texts are automatically preprocessed.
    pub fn predict_with_scores(&self, texts: Vec<String>) -> Vec<(String, f64)> {
        let model = match &self.model {
            Some(m) => m,
            None => return texts.iter().map(|_| ("".to_string(), 0.0)).collect(),
        };

        let texts = self.preprocess_texts(texts);

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

    /// Get the preprocessor (if set).
    #[getter]
    pub fn preprocessor(&self) -> Option<TextPreprocessor> {
        self.preprocessor.clone()
    }

    /// Save model to binary file.
    /// Preprocessor config is saved together with the model.
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

/// Private helper methods (not exposed to Python)
impl TextClassifier {
    /// Preprocess a single text if preprocessor is configured.
    #[inline]
    fn preprocess_text(&self, text: &str) -> String {
        match &self.preprocessor {
            Some(pp) => pp.transform(text),
            None => text.to_string(),
        }
    }

    /// Preprocess a batch of texts if preprocessor is configured.
    #[inline]
    fn preprocess_texts(&self, texts: Vec<String>) -> Vec<String> {
        match &self.preprocessor {
            Some(pp) => pp.transform_batch(&texts),
            None => texts,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::TextPreprocessor;

    fn sample_texts_labels() -> (Vec<String>, Vec<String>) {
        let texts = vec![
            "sản phẩm rất tốt".to_string(),
            "hàng đẹp giá rẻ".to_string(),
            "sản phẩm tốt lắm".to_string(),
            "hàng xấu quá".to_string(),
            "tệ lắm không mua nữa".to_string(),
            "hàng kém chất lượng".to_string(),
        ];
        let labels = vec![
            "positive".to_string(),
            "positive".to_string(),
            "positive".to_string(),
            "negative".to_string(),
            "negative".to_string(),
            "negative".to_string(),
        ];
        (texts, labels)
    }

    #[test]
    fn test_classifier_without_preprocessor() {
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, None);
        clf.fit(texts, labels);
        assert!(clf.is_fitted());
        assert!(clf.preprocessor.is_none());

        let pred = clf.predict("sản phẩm tốt");
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_classifier_with_preprocessor() {
        let pp = TextPreprocessor::default();
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        clf.fit(texts, labels);
        assert!(clf.is_fitted());
        assert!(clf.preprocessor.is_some());

        // Predict with teencode — "ko" should be expanded to "không" by preprocessor
        let pred = clf.predict("sp ko tốt");
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_classifier_predict_batch_with_preprocessor() {
        let pp = TextPreprocessor::default();
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        clf.fit(texts, labels);

        let test_texts = vec!["sp tốt".to_string(), "hàng xấu".to_string()];
        let preds = clf.predict_batch(test_texts);
        assert_eq!(preds.len(), 2);
        assert!(!preds[0].is_empty());
        assert!(!preds[1].is_empty());
    }

    #[test]
    fn test_classifier_predict_with_score_preprocessor() {
        let pp = TextPreprocessor::default();
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        clf.fit(texts, labels);

        let (label, score) = clf.predict_with_score("sp tốt lắm");
        assert!(!label.is_empty());
        assert!(score != 0.0);
    }

    #[test]
    fn test_classifier_predict_with_scores_preprocessor() {
        let pp = TextPreprocessor::default();
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        clf.fit(texts, labels);

        let results = clf.predict_with_scores(vec!["sp tốt".to_string(), "hàng xấu".to_string()]);
        assert_eq!(results.len(), 2);
        for (label, score) in &results {
            assert!(!label.is_empty());
            assert!(*score != 0.0);
        }
    }

    #[test]
    fn test_classifier_serialization_with_preprocessor() {
        let pp = TextPreprocessor::default();
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        clf.fit(texts, labels);

        // Serialize/deserialize via bincode directly (no PyO3 dependency)
        let bytes = bincode::serialize(&clf).unwrap();
        let clf2: TextClassifier = bincode::deserialize(&bytes).unwrap();
        assert!(clf2.is_fitted());
        assert!(clf2.preprocessor.is_some());

        // Both should produce same prediction
        let pred1 = clf.predict("sp ko tốt");
        let pred2 = clf2.predict("sp ko tốt");
        assert_eq!(pred1, pred2);
    }

    #[test]
    fn test_classifier_serialization_without_preprocessor() {
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, None);
        clf.fit(texts, labels);

        let bytes = bincode::serialize(&clf).unwrap();
        let clf2: TextClassifier = bincode::deserialize(&bytes).unwrap();
        assert!(clf2.is_fitted());
        assert!(clf2.preprocessor.is_none());

        let pred1 = clf.predict("sản phẩm tốt");
        let pred2 = clf2.predict("sản phẩm tốt");
        assert_eq!(pred1, pred2);
    }

    #[test]
    fn test_preprocess_text_helper() {
        let pp = TextPreprocessor::default();
        let clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        let result = clf.preprocess_text("Ko đẹp!!!");
        assert_eq!(result, "không NEG_đẹp!");
    }

    #[test]
    fn test_preprocess_text_no_preprocessor() {
        let clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, None);
        let result = clf.preprocess_text("Ko đẹp!!!");
        assert_eq!(result, "Ko đẹp!!!");
    }

    #[test]
    fn test_preprocess_texts_helper() {
        let pp = TextPreprocessor::default();
        let clf = TextClassifier::new(1000, (1, 2), 1, 1.0, 1.0, 1000, 0.1, Some(pp));
        let results = clf.preprocess_texts(vec!["Ko đẹp".to_string(), "SP tốt".to_string()]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "không NEG_đẹp");
        assert!(results[1].contains("sản phẩm"));
    }

    #[test]
    fn test_classifier_classes() {
        let (texts, labels) = sample_texts_labels();
        let mut clf = TextClassifier::default();
        clf.fit(texts, labels);
        let classes = clf.classes();
        assert_eq!(classes.len(), 2);
        assert!(classes.contains(&"positive".to_string()));
        assert!(classes.contains(&"negative".to_string()));
    }

    #[test]
    fn test_classifier_not_fitted() {
        let clf = TextClassifier::default();
        assert!(!clf.is_fitted());
        assert_eq!(clf.predict("test"), "");
        assert_eq!(clf.predict_batch(vec!["test".to_string()]), vec![""]);
        assert_eq!(clf.predict_with_score("test"), ("".to_string(), 0.0));
    }
}
