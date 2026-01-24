//! Logistic Regression Model structure and weight management.
//!
//! This module contains the core LR model representation including:
//! - Feature weights (sparse storage)
//! - Class and feature indices
//! - Logit and softmax computation functions

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Manages the mapping between class label strings and numeric IDs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassIndex {
    /// Maps class label strings to their IDs
    class_to_id: HashMap<String, u32>,

    /// Maps IDs back to class label strings
    id_to_class: Vec<String>,
}

impl ClassIndex {
    /// Create a new empty class index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create an ID for the given class label.
    pub fn get_or_insert(&mut self, class: &str) -> u32 {
        if let Some(&id) = self.class_to_id.get(class) {
            id
        } else {
            let id = self.id_to_class.len() as u32;
            self.class_to_id.insert(class.to_string(), id);
            self.id_to_class.push(class.to_string());
            id
        }
    }

    /// Get the ID for a class label, if it exists.
    pub fn get(&self, class: &str) -> Option<u32> {
        self.class_to_id.get(class).copied()
    }

    /// Get the class label for an ID, if it exists.
    pub fn get_class(&self, id: u32) -> Option<&str> {
        self.id_to_class.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of classes.
    pub fn len(&self) -> usize {
        self.id_to_class.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_class.is_empty()
    }

    /// Get all class labels as a slice.
    pub fn classes(&self) -> &[String] {
        &self.id_to_class
    }

    /// Iterate over all (id, class) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.id_to_class
            .iter()
            .enumerate()
            .map(|(id, class)| (id as u32, class.as_str()))
    }
}

/// Manages the mapping between feature strings and numeric IDs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureIndex {
    /// Maps feature strings to their IDs
    feature_to_id: HashMap<String, u32>,

    /// Maps IDs back to feature strings
    id_to_feature: Vec<String>,
}

impl FeatureIndex {
    /// Create a new empty feature index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create an ID for the given feature.
    pub fn get_or_insert(&mut self, feature: &str) -> u32 {
        if let Some(&id) = self.feature_to_id.get(feature) {
            id
        } else {
            let id = self.id_to_feature.len() as u32;
            self.feature_to_id.insert(feature.to_string(), id);
            self.id_to_feature.push(feature.to_string());
            id
        }
    }

    /// Get the ID for a feature, if it exists.
    pub fn get(&self, feature: &str) -> Option<u32> {
        self.feature_to_id.get(feature).copied()
    }

    /// Get the feature for an ID, if it exists.
    pub fn get_feature(&self, id: u32) -> Option<&str> {
        self.id_to_feature.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of features.
    pub fn len(&self) -> usize {
        self.id_to_feature.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_feature.is_empty()
    }

    /// Iterate over all (id, feature) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.id_to_feature
            .iter()
            .enumerate()
            .map(|(id, feature)| (id as u32, feature.as_str()))
    }
}

/// A Logistic Regression model for multi-class classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRModel {
    /// Number of classes
    pub num_classes: usize,

    /// Number of features
    pub num_features: usize,

    /// Class index (string <-> ID mapping)
    pub classes: ClassIndex,

    /// Feature index (string <-> ID mapping)
    pub features: FeatureIndex,

    /// Sparse weights: maps (feature_id, class_id) -> weight
    weights: HashMap<(u32, u32), f64>,

    /// Per-class bias terms
    biases: Vec<f64>,
}

impl LRModel {
    /// Create a new empty LR model.
    pub fn new() -> Self {
        Self {
            num_classes: 0,
            num_features: 0,
            classes: ClassIndex::new(),
            features: FeatureIndex::new(),
            weights: HashMap::new(),
            biases: Vec::new(),
        }
    }

    /// Create a model with pre-defined classes.
    pub fn with_classes(class_labels: Vec<String>) -> Self {
        let mut model = Self::new();
        for label in class_labels {
            model.classes.get_or_insert(&label);
        }
        model.num_classes = model.classes.len();
        model.biases = vec![0.0; model.num_classes];
        model
    }

    /// Initialize biases for all classes.
    fn initialize_biases(&mut self) {
        if self.biases.len() != self.num_classes {
            self.biases = vec![0.0; self.num_classes];
        }
    }

    /// Get weight for (feature_id, class_id).
    pub fn get_weight(&self, feature_id: u32, class_id: u32) -> f64 {
        self.weights
            .get(&(feature_id, class_id))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set weight for (feature_id, class_id).
    pub fn set_weight(&mut self, feature_id: u32, class_id: u32, weight: f64) {
        if weight.abs() > 1e-10 {
            self.weights.insert((feature_id, class_id), weight);
        } else {
            self.weights.remove(&(feature_id, class_id));
        }
    }

    /// Add to weight for (feature_id, class_id).
    pub fn add_weight(&mut self, feature_id: u32, class_id: u32, delta: f64) {
        let entry = self.weights.entry((feature_id, class_id)).or_insert(0.0);
        *entry += delta;
    }

    /// Get bias for class.
    pub fn get_bias(&self, class_id: u32) -> f64 {
        self.biases.get(class_id as usize).copied().unwrap_or(0.0)
    }

    /// Set bias for class.
    pub fn set_bias(&mut self, class_id: u32, bias: f64) {
        self.initialize_biases();
        if (class_id as usize) < self.biases.len() {
            self.biases[class_id as usize] = bias;
        }
    }

    /// Add to bias for class.
    pub fn add_bias(&mut self, class_id: u32, delta: f64) {
        self.initialize_biases();
        if (class_id as usize) < self.biases.len() {
            self.biases[class_id as usize] += delta;
        }
    }

    /// Compute raw logits (scores) for all classes given feature IDs.
    /// Returns a vector of scores indexed by class ID.
    pub fn compute_logits(&self, feature_ids: &[u32]) -> Vec<f64> {
        let mut logits = self.biases.clone();
        if logits.len() != self.num_classes {
            logits = vec![0.0; self.num_classes];
        }

        for &feature_id in feature_ids {
            for (class_id, logit) in logits.iter_mut().enumerate() {
                *logit += self.get_weight(feature_id, class_id as u32);
            }
        }

        logits
    }

    /// Compute softmax probabilities from logits.
    /// Uses the log-sum-exp trick for numerical stability.
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(logit - max) and sum
        let exp_values: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f64 = exp_values.iter().sum();

        // Normalize
        exp_values.iter().map(|&e| e / sum).collect()
    }

    /// Compute log-softmax for numerical stability in training.
    /// Returns log(softmax(logits)).
    pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute log-sum-exp
        let log_sum_exp = max_logit
            + logits
                .iter()
                .map(|&l| (l - max_logit).exp())
                .sum::<f64>()
                .ln();

        // log_softmax = logit - log_sum_exp
        logits.iter().map(|&l| l - log_sum_exp).collect()
    }

    /// Convert feature strings to IDs, creating new IDs if necessary.
    pub fn features_to_ids(&mut self, features: &[String]) -> Vec<u32> {
        let ids: Vec<u32> = features
            .iter()
            .map(|f| self.features.get_or_insert(f))
            .collect();
        self.num_features = self.features.len();
        ids
    }

    /// Convert feature strings to IDs (read-only, skips unknown features).
    pub fn features_to_ids_readonly(&self, features: &[String]) -> Vec<u32> {
        features
            .iter()
            .filter_map(|f| self.features.get(f))
            .collect()
    }

    /// Convert class string to ID, creating new ID if necessary.
    pub fn class_to_id(&mut self, class: &str) -> u32 {
        let id = self.classes.get_or_insert(class);
        self.num_classes = self.classes.len();
        self.initialize_biases();
        id
    }

    /// Convert class string to ID (read-only).
    pub fn class_to_id_readonly(&self, class: &str) -> Option<u32> {
        self.classes.get(class)
    }

    /// Convert class ID to string.
    pub fn id_to_class(&self, id: u32) -> Option<&str> {
        self.classes.get_class(id)
    }

    /// Get all class labels.
    pub fn get_classes(&self) -> Vec<String> {
        self.classes.classes().to_vec()
    }

    /// Get the number of non-zero weights.
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }

    /// Apply L2 regularization decay to all weights.
    pub fn apply_l2_decay(&mut self, factor: f64) {
        for weight in self.weights.values_mut() {
            *weight *= factor;
        }
        for bias in &mut self.biases {
            *bias *= factor;
        }
    }

    /// Apply L1 regularization (soft thresholding).
    pub fn apply_l1_penalty(&mut self, penalty: f64) {
        self.weights.retain(|_, weight| {
            if *weight > penalty {
                *weight -= penalty;
                true
            } else if *weight < -penalty {
                *weight += penalty;
                true
            } else {
                false
            }
        });
    }

    /// Get the squared L2 norm of all weights.
    pub fn l2_norm_squared(&self) -> f64 {
        let weight_norm: f64 = self.weights.values().map(|w| w * w).sum();
        let bias_norm: f64 = self.biases.iter().map(|b| b * b).sum();
        weight_norm + bias_norm
    }

    /// Get the L1 norm of all weights.
    pub fn l1_norm(&self) -> f64 {
        let weight_norm: f64 = self.weights.values().map(|w| w.abs()).sum();
        let bias_norm: f64 = self.biases.iter().map(|b| b.abs()).sum();
        weight_norm + bias_norm
    }

    /// Iterate over weights.
    pub fn weights_iter(&self) -> impl Iterator<Item = (&(u32, u32), &f64)> {
        self.weights.iter()
    }

    /// Get biases as a slice.
    pub fn biases(&self) -> &[f64] {
        &self.biases
    }
}

impl Default for LRModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_model() {
        let model = LRModel::new();
        assert_eq!(model.num_classes, 0);
        assert_eq!(model.num_features, 0);
    }

    #[test]
    fn test_model_with_classes() {
        let model = LRModel::with_classes(vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]);
        assert_eq!(model.num_classes, 3);
        assert_eq!(model.classes.get("positive"), Some(0));
        assert_eq!(model.classes.get("negative"), Some(1));
        assert_eq!(model.classes.get("neutral"), Some(2));
    }

    #[test]
    fn test_class_index() {
        let mut index = ClassIndex::new();
        let id1 = index.get_or_insert("A");
        let id2 = index.get_or_insert("B");
        let id3 = index.get_or_insert("A");

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3);
        assert_eq!(index.len(), 2);
        assert_eq!(index.get("A"), Some(0));
        assert_eq!(index.get_class(0), Some("A"));
    }

    #[test]
    fn test_feature_index() {
        let mut index = FeatureIndex::new();
        let id1 = index.get_or_insert("word=hello");
        let id2 = index.get_or_insert("word=world");
        let id3 = index.get_or_insert("word=hello");

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3);
        assert_eq!(index.len(), 2);
        assert_eq!(index.get("word=hello"), Some(0));
        assert_eq!(index.get_feature(0), Some("word=hello"));
    }

    #[test]
    fn test_weights() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 1.0);
        model.set_weight(0, 1, -0.5);
        model.set_weight(1, 0, 0.3);

        assert_eq!(model.get_weight(0, 0), 1.0);
        assert_eq!(model.get_weight(0, 1), -0.5);
        assert_eq!(model.get_weight(1, 0), 0.3);
        assert_eq!(model.get_weight(1, 1), 0.0); // Default

        model.add_weight(0, 0, 0.5);
        assert_eq!(model.get_weight(0, 0), 1.5);
    }

    #[test]
    fn test_biases() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_bias(0, 0.5);
        model.set_bias(1, -0.3);

        assert_eq!(model.get_bias(0), 0.5);
        assert_eq!(model.get_bias(1), -0.3);

        model.add_bias(0, 0.1);
        assert_eq!(model.get_bias(0), 0.6);
    }

    #[test]
    fn test_compute_logits() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 1.0); // feature 0 -> class A
        model.set_weight(0, 1, 0.5); // feature 0 -> class B
        model.set_weight(1, 0, 0.3); // feature 1 -> class A
        model.set_bias(0, 0.1);
        model.set_bias(1, 0.2);

        let feature_ids = vec![0, 1];
        let logits = model.compute_logits(&feature_ids);

        // Class A: 1.0 + 0.3 + 0.1 = 1.4
        // Class B: 0.5 + 0.0 + 0.2 = 0.7
        assert!((logits[0] - 1.4).abs() < 1e-10);
        assert!((logits[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = LRModel::softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check that probabilities are ordered correctly
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without the max subtraction trick
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = LRModel::softmax(&logits);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = LRModel::log_softmax(&logits);
        let probs = LRModel::softmax(&logits);

        // log_softmax should equal log(softmax)
        for (lp, p) in log_probs.iter().zip(probs.iter()) {
            assert!((lp - p.ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_l2_regularization() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 1.0);
        model.set_bias(0, 2.0);

        model.apply_l2_decay(0.5);

        assert_eq!(model.get_weight(0, 0), 0.5);
        assert_eq!(model.get_bias(0), 1.0);
    }

    #[test]
    fn test_l1_regularization() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 1.0);
        model.set_weight(0, 1, 0.1); // Will be zeroed
        model.set_weight(1, 0, -0.5);

        model.apply_l1_penalty(0.2);

        assert_eq!(model.get_weight(0, 0), 0.8);
        assert_eq!(model.get_weight(0, 1), 0.0); // Removed
        assert_eq!(model.get_weight(1, 0), -0.3);
    }

    #[test]
    fn test_l2_norm_squared() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 3.0);
        model.set_weight(0, 1, 4.0);

        // 3^2 + 4^2 = 25
        assert!((model.l2_norm_squared() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_l1_norm() {
        let mut model = LRModel::with_classes(vec!["A".to_string(), "B".to_string()]);

        model.set_weight(0, 0, 3.0);
        model.set_weight(0, 1, -4.0);

        // |3| + |-4| = 7
        assert!((model.l1_norm() - 7.0).abs() < 1e-10);
    }
}
