//! CRF Tagger for sequence labeling inference.
//!
//! This module provides Viterbi decoding for finding the most likely
//! label sequence given observations.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::map_clone)]

use super::model::CRFModel;
use super::serialization::{CRFFormat, ModelLoader};
use std::path::Path;

/// CRF Tagger for inference/prediction.
#[derive(Debug, Clone)]
pub struct CRFTagger {
    /// The underlying CRF model
    model: CRFModel,
}

/// Result of Viterbi decoding.
#[derive(Debug, Clone)]
pub struct TaggingResult {
    /// Best label sequence (as IDs)
    pub labels: Vec<u32>,

    /// Score of the best sequence
    pub score: f64,

    /// Marginal probabilities (if computed)
    pub marginals: Option<Vec<Vec<f64>>>,
}

impl CRFTagger {
    /// Create a new tagger with an empty model.
    pub fn new() -> Self {
        Self {
            model: CRFModel::new(),
        }
    }

    /// Create a tagger from an existing model.
    pub fn from_model(model: CRFModel) -> Self {
        Self { model }
    }

    /// Load a model from file.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), String> {
        let loader = ModelLoader::new();
        self.model = loader.load(path, CRFFormat::Auto)?;
        Ok(())
    }

    /// Get a reference to the underlying model.
    pub fn model(&self) -> &CRFModel {
        &self.model
    }

    /// Get a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut CRFModel {
        &mut self.model
    }

    /// Tag a sequence of observations using Viterbi decoding.
    ///
    /// # Arguments
    /// * `features` - A sequence of feature vectors, one per token.
    ///   Each inner vector contains feature strings like "word=hello".
    ///
    /// # Returns
    /// * Vector of label strings
    pub fn tag(&self, features: &[Vec<String>]) -> Vec<String> {
        let result = self.tag_with_score(features);
        result
            .labels
            .iter()
            .map(|&id| self.model.id_to_label(id).unwrap_or("O").to_string())
            .collect()
    }

    /// Tag a sequence and return the score.
    pub fn tag_with_score(&self, features: &[Vec<String>]) -> TaggingResult {
        if features.is_empty() {
            return TaggingResult {
                labels: Vec::new(),
                score: 0.0,
                marginals: None,
            };
        }

        // Convert feature strings to attribute IDs
        let attr_ids: Vec<Vec<u32>> = features
            .iter()
            .map(|token_features| self.model.attrs_to_ids_readonly(token_features))
            .collect();

        self.viterbi(&attr_ids)
    }

    /// Viterbi decoding algorithm.
    ///
    /// For each position t and label y:
    ///   score[t][y] = max_{y'} (score[t-1][y'] + transition[y'][y] + emission[t][y])
    fn viterbi(&self, attr_ids: &[Vec<u32>]) -> TaggingResult {
        let n = attr_ids.len();
        let num_labels = self.model.num_labels;

        if n == 0 || num_labels == 0 {
            return TaggingResult {
                labels: Vec::new(),
                score: 0.0,
                marginals: None,
            };
        }

        // Viterbi score matrix: score[t][y] = best score ending at position t with label y
        let mut score = vec![vec![f64::NEG_INFINITY; num_labels]; n];

        // Backpointer matrix: back[t][y] = best previous label for (t, y)
        let mut back = vec![vec![0u32; num_labels]; n];

        // Initialize: first position
        let emission_0 = self.model.emission_scores(&attr_ids[0]);
        for y in 0..num_labels {
            score[0][y] = emission_0[y];
        }

        // Forward pass: positions 1 to n-1
        for t in 1..n {
            let emission_t = self.model.emission_scores(&attr_ids[t]);

            for y in 0..num_labels {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0u32;

                for y_prev in 0..num_labels {
                    let trans = self.model.get_transition(y_prev as u32, y as u32);
                    let s = score[t - 1][y_prev] + trans;

                    if s > best_score {
                        best_score = s;
                        best_prev = y_prev as u32;
                    }
                }

                score[t][y] = best_score + emission_t[y];
                back[t][y] = best_prev;
            }
        }

        // Find best final label
        let mut best_score = f64::NEG_INFINITY;
        let mut best_label = 0u32;
        for y in 0..num_labels {
            if score[n - 1][y] > best_score {
                best_score = score[n - 1][y];
                best_label = y as u32;
            }
        }

        // Backtrack to find best path
        let mut labels = vec![0u32; n];
        labels[n - 1] = best_label;
        for t in (1..n).rev() {
            labels[t - 1] = back[t][labels[t] as usize];
        }

        TaggingResult {
            labels,
            score: best_score,
            marginals: None,
        }
    }

    /// Compute marginal probabilities using forward-backward algorithm.
    pub fn compute_marginals(&self, features: &[Vec<String>]) -> Vec<Vec<f64>> {
        let attr_ids: Vec<Vec<u32>> = features
            .iter()
            .map(|token_features| self.model.attrs_to_ids_readonly(token_features))
            .collect();

        let (alpha, log_z) = self.forward(&attr_ids);
        let beta = self.backward(&attr_ids);

        self.marginals_from_alpha_beta(&alpha, &beta, log_z)
    }

    /// Forward algorithm for computing alpha values.
    /// Returns (alpha, log_Z) where alpha[t][y] = log(sum of scores of all paths ending at (t, y))
    pub fn forward(&self, attr_ids: &[Vec<u32>]) -> (Vec<Vec<f64>>, f64) {
        let n = attr_ids.len();
        let num_labels = self.model.num_labels;

        if n == 0 || num_labels == 0 {
            return (Vec::new(), 0.0);
        }

        let mut alpha = vec![vec![f64::NEG_INFINITY; num_labels]; n];

        // Initialize
        let emission_0 = self.model.emission_scores(&attr_ids[0]);
        for y in 0..num_labels {
            alpha[0][y] = emission_0[y];
        }

        // Forward pass
        for t in 1..n {
            let emission_t = self.model.emission_scores(&attr_ids[t]);

            for y in 0..num_labels {
                let mut log_sum = f64::NEG_INFINITY;

                for y_prev in 0..num_labels {
                    let trans = self.model.get_transition(y_prev as u32, y as u32);
                    let score = alpha[t - 1][y_prev] + trans;
                    log_sum = log_sum_exp(log_sum, score);
                }

                alpha[t][y] = log_sum + emission_t[y];
            }
        }

        // Compute log Z (partition function)
        let mut log_z = f64::NEG_INFINITY;
        for y in 0..num_labels {
            log_z = log_sum_exp(log_z, alpha[n - 1][y]);
        }

        (alpha, log_z)
    }

    /// Backward algorithm for computing beta values.
    /// Returns beta where beta[t][y] = log(sum of scores of all paths starting from (t, y))
    pub fn backward(&self, attr_ids: &[Vec<u32>]) -> Vec<Vec<f64>> {
        let n = attr_ids.len();
        let num_labels = self.model.num_labels;

        if n == 0 || num_labels == 0 {
            return Vec::new();
        }

        let mut beta = vec![vec![f64::NEG_INFINITY; num_labels]; n];

        // Initialize: beta[n-1][y] = 0 (log(1))
        for y in 0..num_labels {
            beta[n - 1][y] = 0.0;
        }

        // Backward pass
        for t in (0..n - 1).rev() {
            let emission_next = self.model.emission_scores(&attr_ids[t + 1]);

            for y in 0..num_labels {
                let mut log_sum = f64::NEG_INFINITY;

                for y_next in 0..num_labels {
                    let trans = self.model.get_transition(y as u32, y_next as u32);
                    let score = trans + emission_next[y_next] + beta[t + 1][y_next];
                    log_sum = log_sum_exp(log_sum, score);
                }

                beta[t][y] = log_sum;
            }
        }

        beta
    }

    /// Compute marginal probabilities from alpha and beta.
    fn marginals_from_alpha_beta(
        &self,
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
        log_z: f64,
    ) -> Vec<Vec<f64>> {
        let n = alpha.len();
        let num_labels = self.model.num_labels;

        let mut marginals = vec![vec![0.0; num_labels]; n];

        for t in 0..n {
            for y in 0..num_labels {
                marginals[t][y] = (alpha[t][y] + beta[t][y] - log_z).exp();
            }
        }

        marginals
    }

    /// Compute the score of a given label sequence.
    pub fn score_sequence(&self, features: &[Vec<String>], labels: &[u32]) -> f64 {
        let attr_ids: Vec<Vec<u32>> = features
            .iter()
            .map(|token_features| self.model.attrs_to_ids_readonly(token_features))
            .collect();

        self.sequence_score(&attr_ids, labels)
    }

    /// Compute the score of a label sequence (internal).
    pub fn sequence_score(&self, attr_ids: &[Vec<u32>], labels: &[u32]) -> f64 {
        if attr_ids.is_empty() || labels.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        // Emission scores
        for (t, label) in labels.iter().enumerate() {
            score += self.model.emission_score(&attr_ids[t], *label);
        }

        // Transition scores
        for t in 1..labels.len() {
            score += self.model.get_transition(labels[t - 1], labels[t]);
        }

        score
    }

    /// Get the number of labels.
    pub fn num_labels(&self) -> usize {
        self.model.num_labels
    }

    /// Get the list of labels.
    pub fn labels(&self) -> Vec<String> {
        self.model
            .labels
            .labels()
            .iter()
            .map(|s| s.clone())
            .collect()
    }
}

impl Default for CRFTagger {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute log(exp(a) + exp(b)) in a numerically stable way.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    fn create_test_model() -> CRFModel {
        let mut model = CRFModel::with_labels(vec![
            "B-PER".to_string(),
            "I-PER".to_string(),
            "O".to_string(),
        ]);

        // Add some state features
        let attr_word_john = model.attributes.get_or_insert("word=John");
        let attr_word_smith = model.attributes.get_or_insert("word=Smith");
        let attr_word_is = model.attributes.get_or_insert("word=is");

        model.num_attributes = model.attributes.len();

        // "John" -> B-PER
        model.set_state_weight(attr_word_john, 0, 2.0);
        // "Smith" -> I-PER
        model.set_state_weight(attr_word_smith, 1, 2.0);
        // "is" -> O
        model.set_state_weight(attr_word_is, 2, 1.5);

        // Transitions
        model.set_transition(0, 1, 1.0); // B-PER -> I-PER
        model.set_transition(0, 2, 0.5); // B-PER -> O
        model.set_transition(1, 2, 0.5); // I-PER -> O
        model.set_transition(2, 0, 0.3); // O -> B-PER
        model.set_transition(2, 2, 0.5); // O -> O

        model
    }

    #[test]
    fn test_tagger_creation() {
        let tagger = CRFTagger::new();
        assert_eq!(tagger.num_labels(), 0);

        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);
        assert_eq!(tagger.num_labels(), 3);
    }

    #[test]
    fn test_viterbi_simple() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features = vec![
            vec!["word=John".to_string()],
            vec!["word=Smith".to_string()],
            vec!["word=is".to_string()],
        ];

        let result = tagger.tag_with_score(&features);
        assert_eq!(result.labels.len(), 3);

        // Should predict B-PER, I-PER, O
        let labels = tagger.tag(&features);
        assert_eq!(labels, vec!["B-PER", "I-PER", "O"]);
    }

    #[test]
    fn test_empty_sequence() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features: Vec<Vec<String>> = vec![];
        let result = tagger.tag_with_score(&features);

        assert!(result.labels.is_empty());
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_single_token() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features = vec![vec!["word=John".to_string()]];
        let labels = tagger.tag(&features);

        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], "B-PER");
    }

    #[test]
    fn test_forward_backward() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features = vec![
            vec!["word=John".to_string()],
            vec!["word=Smith".to_string()],
        ];

        let attr_ids: Vec<Vec<u32>> = features
            .iter()
            .map(|f| tagger.model.attrs_to_ids_readonly(f))
            .collect();

        let (alpha, _log_z) = tagger.forward(&attr_ids);
        let beta = tagger.backward(&attr_ids);

        assert_eq!(alpha.len(), 2);
        assert_eq!(beta.len(), 2);
        assert_eq!(alpha[0].len(), 3);
        assert_eq!(beta[0].len(), 3);
    }

    #[test]
    fn test_marginals() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features = vec![
            vec!["word=John".to_string()],
            vec!["word=Smith".to_string()],
        ];

        let marginals = tagger.compute_marginals(&features);

        assert_eq!(marginals.len(), 2);

        // Marginals should sum to 1 at each position
        for m in &marginals {
            let sum: f64 = m.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Marginals should sum to 1");
        }
    }

    #[test]
    fn test_log_sum_exp() {
        // Test with normal values
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);

        // Test with negative infinity
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(log_sum_exp(0.0, f64::NEG_INFINITY), 0.0);
        assert_eq!(
            log_sum_exp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn test_sequence_score() {
        let model = create_test_model();
        let tagger = CRFTagger::from_model(model);

        let features = vec![
            vec!["word=John".to_string()],
            vec!["word=Smith".to_string()],
        ];

        let labels = vec![0, 1]; // B-PER, I-PER
        let score = tagger.score_sequence(&features, &labels);

        // Score = emission(John, B-PER) + transition(B-PER, I-PER) + emission(Smith, I-PER)
        // = 2.0 + 1.0 + 2.0 = 5.0
        assert!((score - 5.0).abs() < 1e-6);
    }
}
