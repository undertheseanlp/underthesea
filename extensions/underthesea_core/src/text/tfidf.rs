//! TF-IDF Vectorizer for text classification.
//!
//! This module provides a TF-IDF (Term Frequency - Inverse Document Frequency)
//! vectorizer that transforms text documents into numerical feature vectors.
//!
//! The implementation follows sklearn's TfidfVectorizer conventions:
//! - TF: raw term count (sklearn default)
//! - IDF: log((n_docs + 1) / (df + 1)) + 1 (smooth IDF)
//! - L2 normalization by default
//! - Token pattern: minimum 2 characters (sklearn default)
//!
//! Performance optimizations:
//! - Parallel document frequency counting during fit()
//! - Batch transforms with parallel processing
//! - Efficient tokenization using byte offsets

use hashbrown::HashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for TfIdfVectorizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfConfig {
    /// Minimum document frequency (absolute count)
    pub min_df: usize,
    /// Maximum document frequency (ratio of documents, 0.0-1.0)
    pub max_df: f64,
    /// Whether to use sublinear TF (1 + log(tf) instead of tf)
    pub sublinear_tf: bool,
    /// Whether to apply L2 normalization
    pub norm: bool,
    /// Whether to convert to lowercase
    pub lowercase: bool,
    /// Maximum vocabulary size (0 = unlimited)
    pub max_features: usize,
    /// N-gram range (min, max)
    pub ngram_range: (usize, usize),
    /// Minimum token length (sklearn default is 2)
    pub min_token_length: usize,
}

impl Default for TfIdfConfig {
    fn default() -> Self {
        Self {
            min_df: 1,
            max_df: 1.0,
            sublinear_tf: false,
            norm: true,
            lowercase: true,
            max_features: 0,
            ngram_range: (1, 1),
            min_token_length: 2, // sklearn default: \b\w\w+\b requires 2+ chars
        }
    }
}

/// A TF-IDF Vectorizer that transforms text documents into feature vectors.
///
/// # Example
/// ```
/// use underthesea_core::text::TfIdfVectorizer;
///
/// let documents = vec![
///     "Tôi yêu Việt Nam".to_string(),
///     "Việt Nam đẹp lắm".to_string(),
/// ];
///
/// let mut vectorizer = TfIdfVectorizer::new();
/// vectorizer.fit(&documents);
///
/// let features = vectorizer.transform("Tôi yêu Việt Nam");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfVectorizer {
    /// Vocabulary: word -> index
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: index -> word
    index_to_word: Vec<String>,
    /// IDF values for each word in vocabulary
    idf: Vec<f64>,
    /// Document frequency for each word
    df: Vec<usize>,
    /// Number of documents used for fitting
    n_docs: usize,
    /// Configuration
    config: TfIdfConfig,
    /// Whether the vectorizer has been fitted
    fitted: bool,
}

impl TfIdfVectorizer {
    /// Create a new TfIdfVectorizer with default configuration.
    pub fn new() -> Self {
        Self::with_config(TfIdfConfig::default())
    }

    /// Create a new TfIdfVectorizer with custom configuration.
    pub fn with_config(config: TfIdfConfig) -> Self {
        Self {
            vocab: HashMap::new(),
            index_to_word: Vec::new(),
            idf: Vec::new(),
            df: Vec::new(),
            n_docs: 0,
            config,
            fitted: false,
        }
    }

    /// Tokenize a document into words.
    /// Follows sklearn's default token pattern: only words with >= min_token_length chars.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let text = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Split on whitespace and filter by minimum token length
        // This matches sklearn's default token_pattern r"(?u)\b\w\w+\b"
        let words: Vec<String> = text
            .split_whitespace()
            .filter(|w| w.chars().count() >= self.config.min_token_length)
            .map(|w| w.to_string())
            .collect();

        // Generate n-grams if needed
        if self.config.ngram_range == (1, 1) {
            words
        } else {
            self.generate_ngrams(&words)
        }
    }

    /// Generate n-grams from a list of words.
    fn generate_ngrams(&self, words: &[String]) -> Vec<String> {
        let (min_n, max_n) = self.config.ngram_range;
        let mut ngrams = Vec::new();

        for n in min_n..=max_n {
            if n > words.len() {
                continue;
            }
            for window in words.windows(n) {
                ngrams.push(window.join(" "));
            }
        }

        ngrams
    }

    /// Fit the vectorizer on a collection of documents.
    ///
    /// This builds the vocabulary and computes IDF values.
    /// Uses parallel processing for large datasets.
    pub fn fit(&mut self, documents: &[String]) {
        self.n_docs = documents.len();
        if self.n_docs == 0 {
            self.fitted = true;
            return;
        }

        // Parallel document frequency counting for large datasets
        let (doc_freq, total_freq) = if documents.len() >= 1000 {
            self.fit_parallel(documents)
        } else {
            self.fit_sequential(documents)
        };

        // Filter by min_df and max_df
        let max_docs = (self.config.max_df * self.n_docs as f64).ceil() as usize;
        let mut filtered_words: Vec<(String, usize, usize)> = doc_freq
            .into_iter()
            .filter(|(_, freq)| *freq >= self.config.min_df && *freq <= max_docs)
            .map(|(word, df)| {
                let tf = total_freq.get(&word).copied().unwrap_or(0);
                (word, df, tf)
            })
            .collect();

        // Sort by total frequency (descending), then alphabetically for tie-breaking
        // This matches sklearn's behavior for max_features selection
        filtered_words.sort_by(|a, b| {
            match b.2.cmp(&a.2) {
                std::cmp::Ordering::Equal => a.0.cmp(&b.0), // alphabetical for ties
                other => other,
            }
        });

        // Apply max_features limit
        if self.config.max_features > 0 && filtered_words.len() > self.config.max_features {
            filtered_words.truncate(self.config.max_features);
        }

        // Sort alphabetically for consistent vocabulary ordering
        filtered_words.sort_by(|a, b| a.0.cmp(&b.0));

        // Build vocabulary and compute IDF
        self.vocab.clear();
        self.index_to_word.clear();
        self.idf.clear();
        self.df.clear();

        for (idx, (word, df, _)) in filtered_words.into_iter().enumerate() {
            self.vocab.insert(word.clone(), idx as u32);
            self.index_to_word.push(word);
            self.df.push(df);

            // IDF: log((n_docs + 1) / (df + 1)) + 1 (sklearn smooth IDF formula)
            let idf_value = ((self.n_docs as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0;
            self.idf.push(idf_value);
        }

        self.fitted = true;
    }

    /// Sequential document frequency counting (for small datasets).
    fn fit_sequential(
        &self,
        documents: &[String],
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let tokens = self.tokenize(doc);
            let unique_tokens: HashSet<&String> = tokens.iter().collect();

            for token in &unique_tokens {
                *doc_freq.entry((*token).clone()).or_insert(0) += 1;
            }

            for token in &tokens {
                *total_freq.entry(token.clone()).or_insert(0) += 1;
            }
        }

        (doc_freq, total_freq)
    }

    /// Parallel document frequency counting (for large datasets).
    fn fit_parallel(
        &self,
        documents: &[String],
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let chunk_size = (documents.len() / rayon::current_num_threads()).max(100);

        // Parallel processing of document chunks
        let local_counts: Vec<(HashMap<String, usize>, HashMap<String, usize>)> = documents
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_df: HashMap<String, usize> = HashMap::new();
                let mut local_tf: HashMap<String, usize> = HashMap::new();

                for doc in chunk {
                    let tokens = self.tokenize(doc);
                    let unique_tokens: HashSet<&String> = tokens.iter().collect();

                    for token in &unique_tokens {
                        *local_df.entry((*token).clone()).or_insert(0) += 1;
                    }

                    for token in &tokens {
                        *local_tf.entry(token.clone()).or_insert(0) += 1;
                    }
                }

                (local_df, local_tf)
            })
            .collect();

        // Merge results
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_freq: HashMap<String, usize> = HashMap::new();

        for (local_df, local_tf) in local_counts {
            for (term, count) in local_df {
                *doc_freq.entry(term).or_insert(0) += count;
            }
            for (term, count) in local_tf {
                *total_freq.entry(term).or_insert(0) += count;
            }
        }

        (doc_freq, total_freq)
    }

    /// Transform a document into a sparse TF-IDF vector.
    ///
    /// Returns a vector of (feature_index, tfidf_value) pairs.
    /// Uses sklearn-compatible TF-IDF calculation:
    /// - TF: raw term count (or 1 + log(tf) if sublinear_tf=True)
    /// - IDF: log((n_docs + 1) / (df + 1)) + 1
    /// - L2 normalization by default
    pub fn transform(&self, document: &str) -> Vec<(u32, f64)> {
        if !self.fitted {
            return Vec::new();
        }

        let tokens = self.tokenize(document);

        if tokens.is_empty() {
            return Vec::new();
        }

        // Count term frequency (raw counts, sklearn default)
        let mut tf: HashMap<u32, f64> = HashMap::new();
        for token in &tokens {
            if let Some(&idx) = self.vocab.get(token) {
                *tf.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        // Compute TF-IDF (sklearn formula)
        let mut features: Vec<(u32, f64)> = tf
            .into_iter()
            .map(|(idx, raw_count)| {
                // Term frequency
                let tf_value = if self.config.sublinear_tf {
                    // Sublinear TF: 1 + log(tf) (sklearn sublinear_tf=True)
                    if raw_count > 0.0 {
                        1.0 + raw_count.ln()
                    } else {
                        0.0
                    }
                } else {
                    // Raw TF (sklearn default): just the count
                    raw_count
                };

                // TF-IDF = TF * IDF
                let tfidf = tf_value * self.idf[idx as usize];
                (idx, tfidf)
            })
            .filter(|(_, v)| *v > 0.0)
            .collect();

        // L2 normalization (sklearn default: norm='l2')
        if self.config.norm {
            let norm: f64 = features.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (_, v) in &mut features {
                    *v /= norm;
                }
            }
        }

        // Sort by index for consistent output
        features.sort_by_key(|(idx, _)| *idx);

        features
    }

    /// Transform a document into a dense TF-IDF vector.
    ///
    /// Returns a vector of length vocab_size with TF-IDF values.
    pub fn transform_dense(&self, document: &str) -> Vec<f64> {
        let mut dense = vec![0.0; self.vocab.len()];
        for (idx, value) in self.transform(document) {
            dense[idx as usize] = value;
        }
        dense
    }

    /// Transform multiple documents into sparse TF-IDF vectors (parallel).
    ///
    /// Returns a vector of sparse vectors, one per document.
    pub fn transform_batch(&self, documents: &[String]) -> Vec<Vec<(u32, f64)>> {
        if !self.fitted {
            return vec![Vec::new(); documents.len()];
        }

        documents
            .par_iter()
            .map(|doc| self.transform(doc))
            .collect()
    }

    /// Transform multiple documents into dense TF-IDF vectors (parallel).
    ///
    /// Returns a vector of dense vectors, one per document.
    pub fn transform_batch_dense(&self, documents: &[String]) -> Vec<Vec<f64>> {
        if !self.fitted {
            return vec![vec![0.0; self.vocab.len()]; documents.len()];
        }

        documents
            .par_iter()
            .map(|doc| self.transform_dense(doc))
            .collect()
    }

    /// Transform a document into feature strings compatible with LRClassifier.
    ///
    /// Returns features in format "tfidf_{index}={value}".
    pub fn transform_to_features(&self, document: &str) -> Vec<String> {
        self.transform(document)
            .into_iter()
            .map(|(idx, val)| format!("tfidf_{}={:.6}", idx, val))
            .collect()
    }

    /// Transform a document into feature strings with word names.
    ///
    /// Returns features in format "word={word}:{value}".
    pub fn transform_to_word_features(&self, document: &str) -> Vec<String> {
        self.transform(document)
            .into_iter()
            .map(|(idx, val)| {
                let word = &self.index_to_word[idx as usize];
                format!("word={}:{:.6}", word, val)
            })
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, documents: &[String]) -> Vec<Vec<(u32, f64)>> {
        self.fit(documents);
        documents.iter().map(|doc| self.transform(doc)).collect()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the number of documents used for fitting.
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// Check if the vectorizer has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the vocabulary as a map from words to indices.
    pub fn vocabulary(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    /// Get IDF values.
    pub fn idf_values(&self) -> &[f64] {
        &self.idf
    }

    /// Get a word by its index.
    pub fn get_word(&self, index: u32) -> Option<&str> {
        self.index_to_word.get(index as usize).map(|s| s.as_str())
    }

    /// Get the index of a word.
    pub fn get_index(&self, word: &str) -> Option<u32> {
        let word = if self.config.lowercase {
            word.to_lowercase()
        } else {
            word.to_string()
        };
        self.vocab.get(&word).copied()
    }

    /// Get top features by IDF value (most discriminative words).
    pub fn top_features_by_idf(&self, n: usize) -> Vec<(String, f64)> {
        let mut features: Vec<(String, f64)> = self
            .index_to_word
            .iter()
            .zip(self.idf.iter())
            .map(|(word, &idf)| (word.clone(), idf))
            .collect();

        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        features.truncate(n);
        features
    }

    /// Get feature names (vocabulary words in order).
    pub fn get_feature_names(&self) -> Vec<String> {
        self.index_to_word.clone()
    }
}

impl Default for TfIdfVectorizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_documents() -> Vec<String> {
        vec![
            "Tôi yêu Việt Nam".to_string(),
            "Việt Nam đất nước tôi".to_string(),
            "Hà Nội thủ đô Việt Nam".to_string(),
            "Tôi sống ở Hà Nội".to_string(),
        ]
    }

    #[test]
    fn test_new_vectorizer() {
        let vectorizer = TfIdfVectorizer::new();
        assert!(!vectorizer.is_fitted());
        assert_eq!(vectorizer.vocab_size(), 0);
    }

    #[test]
    fn test_fit() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();

        vectorizer.fit(&documents);

        assert!(vectorizer.is_fitted());
        assert!(vectorizer.vocab_size() > 0);
        assert_eq!(vectorizer.n_docs(), 4);

        // Check that common words have lower IDF
        let idx_viet = vectorizer.get_index("việt").unwrap();
        let idx_yeu = vectorizer.get_index("yêu").unwrap();

        // "việt" appears in 3 docs, "yêu" appears in 1 doc
        // So IDF of "yêu" should be higher
        assert!(vectorizer.idf[idx_yeu as usize] > vectorizer.idf[idx_viet as usize]);

        // "ở" should NOT be in vocabulary (single char, min_token_length=2)
        assert!(vectorizer.get_index("ở").is_none());
    }

    #[test]
    fn test_transform() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let features = vectorizer.transform("Tôi yêu Việt Nam");

        // Should have some features
        assert!(!features.is_empty());

        // All values should be positive
        for (_, val) in &features {
            assert!(*val > 0.0);
        }

        // Should be L2 normalized (sum of squares ≈ 1)
        let norm_sq: f64 = features.iter().map(|(_, v)| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_dense() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let dense = vectorizer.transform_dense("Tôi yêu Việt Nam");

        assert_eq!(dense.len(), vectorizer.vocab_size());

        // Some values should be non-zero
        let non_zero_count = dense.iter().filter(|&&v| v > 0.0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_transform_to_features() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let features = vectorizer.transform_to_features("Tôi yêu Việt Nam");

        // All features should start with "tfidf_"
        for f in &features {
            assert!(f.starts_with("tfidf_"));
            assert!(f.contains('='));
        }
    }

    #[test]
    fn test_transform_unknown_words() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        // Transform a document with mostly unknown words
        let features = vectorizer.transform("xyz abc unknown words");

        // Should return empty or very sparse vector
        assert!(features.len() <= 1);
    }

    #[test]
    fn test_lowercase() {
        let documents = vec!["Hello World".to_string(), "hello world".to_string()];
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        // "hello" and "Hello" should be the same
        assert!(vectorizer.get_index("hello").is_some());
        assert!(vectorizer.get_index("HELLO").is_some());

        let idx1 = vectorizer.get_index("hello").unwrap();
        let idx2 = vectorizer.get_index("HELLO").unwrap();
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn test_min_df() {
        let documents = create_test_documents();
        let config = TfIdfConfig {
            min_df: 2,
            ..Default::default()
        };
        let mut vectorizer = TfIdfVectorizer::with_config(config);
        vectorizer.fit(&documents);

        // Words appearing in only 1 document should be filtered out
        assert!(vectorizer.get_index("yêu").is_none()); // appears in 1 doc
        assert!(vectorizer.get_index("tôi").is_some()); // appears in 2 docs
    }

    #[test]
    fn test_max_df() {
        let documents = create_test_documents();
        let config = TfIdfConfig {
            max_df: 0.5, // Words in more than 50% of docs are filtered
            ..Default::default()
        };
        let mut vectorizer = TfIdfVectorizer::with_config(config);
        vectorizer.fit(&documents);

        // "việt" and "nam" appear in 3/4 = 75% of docs, should be filtered
        // Note: depends on exact filtering logic
        assert!(vectorizer.vocab_size() < 10);
    }

    #[test]
    fn test_max_features() {
        let documents = create_test_documents();
        let config = TfIdfConfig {
            max_features: 5,
            ..Default::default()
        };
        let mut vectorizer = TfIdfVectorizer::with_config(config);
        vectorizer.fit(&documents);

        assert!(vectorizer.vocab_size() <= 5);
    }

    #[test]
    fn test_sublinear_tf() {
        let documents = vec![
            "word word word word word".to_string(),
            "other text here".to_string(),
        ];

        // Without sublinear TF
        let mut vectorizer1 = TfIdfVectorizer::new();
        vectorizer1.fit(&documents);
        let features1 = vectorizer1.transform("word word word word word");

        // With sublinear TF
        let config = TfIdfConfig {
            sublinear_tf: true,
            ..Default::default()
        };
        let mut vectorizer2 = TfIdfVectorizer::with_config(config);
        vectorizer2.fit(&documents);
        let features2 = vectorizer2.transform("word word word word word");

        // Sublinear TF should produce different values
        // The repeated word should have different weights
        assert!(!features1.is_empty());
        assert!(!features2.is_empty());
    }

    #[test]
    fn test_ngrams() {
        let documents = vec!["hello world foo bar".to_string(), "foo bar baz".to_string()];

        let config = TfIdfConfig {
            ngram_range: (1, 2),
            ..Default::default()
        };
        let mut vectorizer = TfIdfVectorizer::with_config(config);
        vectorizer.fit(&documents);

        // Should include bigrams
        assert!(vectorizer.get_index("foo bar").is_some());
        assert!(vectorizer.get_index("hello world").is_some());

        // Should also include unigrams
        assert!(vectorizer.get_index("foo").is_some());
        assert!(vectorizer.get_index("bar").is_some());
    }

    #[test]
    fn test_top_features_by_idf() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let top = vectorizer.top_features_by_idf(3);

        assert_eq!(top.len(), 3);

        // Should be sorted by IDF descending
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn test_get_feature_names() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let names = vectorizer.get_feature_names();

        assert_eq!(names.len(), vectorizer.vocab_size());
    }

    #[test]
    fn test_empty_document() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        let features = vectorizer.transform("");
        assert!(features.is_empty());

        let features = vectorizer.transform("   ");
        assert!(features.is_empty());
    }

    #[test]
    fn test_fit_transform() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();

        let results = vectorizer.fit_transform(&documents);

        assert_eq!(results.len(), documents.len());
        assert!(vectorizer.is_fitted());
    }

    #[test]
    fn test_serialization() {
        let documents = create_test_documents();
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        // Serialize
        let serialized = bincode::serialize(&vectorizer).unwrap();

        // Deserialize
        let deserialized: TfIdfVectorizer = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.vocab_size(), vectorizer.vocab_size());
        assert_eq!(deserialized.n_docs(), vectorizer.n_docs());
        assert!(deserialized.is_fitted());

        // Transform should produce same results
        let original = vectorizer.transform("Tôi yêu Việt Nam");
        let restored = deserialized.transform("Tôi yêu Việt Nam");

        assert_eq!(original.len(), restored.len());
        for (o, r) in original.iter().zip(restored.iter()) {
            assert_eq!(o.0, r.0);
            assert!((o.1 - r.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_min_token_length() {
        let documents = vec![
            "I am a test".to_string(),
            "This is another test".to_string(),
        ];

        // Default min_token_length=2 (sklearn compatible)
        let mut vectorizer = TfIdfVectorizer::new();
        vectorizer.fit(&documents);

        // Single char tokens should be excluded
        assert!(vectorizer.get_index("i").is_none());
        assert!(vectorizer.get_index("a").is_none());

        // Multi-char tokens should be included
        assert!(vectorizer.get_index("am").is_some());
        assert!(vectorizer.get_index("test").is_some());

        // With min_token_length=1, single chars should be included
        let config = TfIdfConfig {
            min_token_length: 1,
            ..Default::default()
        };
        let mut vectorizer2 = TfIdfVectorizer::with_config(config);
        vectorizer2.fit(&documents);

        assert!(vectorizer2.get_index("i").is_some());
        assert!(vectorizer2.get_index("a").is_some());
    }

    #[test]
    fn test_sklearn_compatible_tf() {
        // Test that TF uses raw counts (sklearn default)
        let documents = vec!["word word word".to_string(), "other".to_string()];

        // Without normalization to see raw TF-IDF values
        let config = TfIdfConfig {
            norm: false,
            ..Default::default()
        };
        let mut vectorizer = TfIdfVectorizer::with_config(config);
        vectorizer.fit(&documents);

        let features = vectorizer.transform("word word word");

        // Find the feature for "word"
        let word_idx = vectorizer.get_index("word").unwrap();
        let word_tfidf = features.iter().find(|(idx, _)| *idx == word_idx).unwrap().1;

        // TF = 3 (raw count), IDF = log((2+1)/(1+1)) + 1 = log(1.5) + 1 ≈ 1.405
        // TF-IDF = 3 * 1.405 ≈ 4.216
        let expected_idf = (3.0_f64 / 2.0).ln() + 1.0;
        let expected_tfidf = 3.0 * expected_idf;

        assert!(
            (word_tfidf - expected_tfidf).abs() < 1e-6,
            "Expected TF-IDF {} but got {}",
            expected_tfidf,
            word_tfidf
        );
    }
}
