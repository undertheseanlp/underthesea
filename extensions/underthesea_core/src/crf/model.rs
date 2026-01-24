//! CRF Model structure and weight management.
//!
//! This module contains the core CRF model representation including:
//! - Feature weights (state and transition)
//! - Label and attribute indices
//! - Score computation functions

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::features::{AttributeIndex, FeatureFunction, FeatureType, LabelIndex};

/// A CRF model containing all weights and indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRFModel {
    /// Number of labels
    pub num_labels: usize,

    /// Number of attributes
    pub num_attributes: usize,

    /// Label index (string <-> ID mapping)
    pub labels: LabelIndex,

    /// Attribute index (string <-> ID mapping)
    pub attributes: AttributeIndex,

    /// State features: maps (attr_id, label_id) -> weight
    /// These are emission/observation features
    state_weights: HashMap<(u32, u32), f64>,

    /// Transition features: maps (from_label, to_label) -> weight
    /// Stored as a flattened matrix for efficiency
    transition_weights: Vec<f64>,

    /// All features (for serialization and iteration)
    features: Vec<FeatureFunction>,
}

impl CRFModel {
    /// Create a new empty CRF model.
    pub fn new() -> Self {
        Self {
            num_labels: 0,
            num_attributes: 0,
            labels: LabelIndex::new(),
            attributes: AttributeIndex::new(),
            state_weights: HashMap::new(),
            transition_weights: Vec::new(),
            features: Vec::new(),
        }
    }

    /// Create a model with pre-defined labels.
    pub fn with_labels(labels: Vec<String>) -> Self {
        let mut model = Self::new();
        for label in labels {
            model.labels.get_or_insert(&label);
        }
        model.num_labels = model.labels.len();
        model.initialize_transition_weights();
        model
    }

    /// Initialize transition weight matrix.
    fn initialize_transition_weights(&mut self) {
        let n = self.num_labels;
        self.transition_weights = vec![0.0; n * n];
    }

    /// Get transition weight from label i to label j.
    pub fn get_transition(&self, from_label: u32, to_label: u32) -> f64 {
        let idx = from_label as usize * self.num_labels + to_label as usize;
        self.transition_weights.get(idx).copied().unwrap_or(0.0)
    }

    /// Set transition weight from label i to label j.
    pub fn set_transition(&mut self, from_label: u32, to_label: u32, weight: f64) {
        if self.transition_weights.is_empty() {
            self.initialize_transition_weights();
        }
        let idx = from_label as usize * self.num_labels + to_label as usize;
        if idx < self.transition_weights.len() {
            self.transition_weights[idx] = weight;
        }
    }

    /// Add to transition weight (for training updates).
    pub fn add_transition(&mut self, from_label: u32, to_label: u32, delta: f64) {
        if self.transition_weights.is_empty() {
            self.initialize_transition_weights();
        }
        let idx = from_label as usize * self.num_labels + to_label as usize;
        if idx < self.transition_weights.len() {
            self.transition_weights[idx] += delta;
        }
    }

    /// Get state weight for (attribute, label).
    pub fn get_state_weight(&self, attr_id: u32, label_id: u32) -> f64 {
        self.state_weights
            .get(&(attr_id, label_id))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set state weight for (attribute, label).
    pub fn set_state_weight(&mut self, attr_id: u32, label_id: u32, weight: f64) {
        if weight.abs() > 1e-10 {
            self.state_weights.insert((attr_id, label_id), weight);
        } else {
            self.state_weights.remove(&(attr_id, label_id));
        }
    }

    /// Add to state weight (for training updates).
    pub fn add_state_weight(&mut self, attr_id: u32, label_id: u32, delta: f64) {
        let entry = self.state_weights.entry((attr_id, label_id)).or_insert(0.0);
        *entry += delta;
    }

    /// Compute the emission score for a token with given attributes and label.
    /// This sums up all state feature weights that match the attributes.
    pub fn emission_score(&self, attr_ids: &[u32], label_id: u32) -> f64 {
        let mut score = 0.0;
        for &attr_id in attr_ids {
            score += self.get_state_weight(attr_id, label_id);
        }
        score
    }

    /// Compute scores for all labels given attributes.
    /// Returns a vector of scores indexed by label ID.
    pub fn emission_scores(&self, attr_ids: &[u32]) -> Vec<f64> {
        let mut scores = vec![0.0; self.num_labels];
        for &attr_id in attr_ids {
            for label_id in 0..self.num_labels {
                scores[label_id] += self.get_state_weight(attr_id, label_id as u32);
            }
        }
        scores
    }

    /// Get all transition weights as a slice.
    pub fn transition_weights(&self) -> &[f64] {
        &self.transition_weights
    }

    /// Get the number of state features.
    pub fn num_state_features(&self) -> usize {
        self.state_weights.len()
    }

    /// Get the number of transition features.
    pub fn num_transition_features(&self) -> usize {
        self.transition_weights.len()
    }

    /// Add a new feature to the model.
    pub fn add_feature(&mut self, feature: FeatureFunction) {
        match feature.feature_type {
            FeatureType::State | FeatureType::Unigram | FeatureType::Bigram => {
                self.set_state_weight(feature.source, feature.target, feature.weight);
            }
            FeatureType::Transition => {
                self.set_transition(feature.source, feature.target, feature.weight);
            }
            _ => {
                // Custom features handled as state features
                self.set_state_weight(feature.source, feature.target, feature.weight);
            }
        }
        self.features.push(feature);
    }

    /// Get all features.
    pub fn features(&self) -> &[FeatureFunction] {
        &self.features
    }

    /// Build features list from current weights (for serialization).
    pub fn build_features_list(&mut self) {
        self.features.clear();
        let mut id = 0u32;

        // Add state features
        for (&(attr_id, label_id), &weight) in &self.state_weights {
            self.features
                .push(FeatureFunction::new_state(id, attr_id, label_id, weight));
            id += 1;
        }

        // Add transition features
        for from_label in 0..self.num_labels {
            for to_label in 0..self.num_labels {
                let weight = self.get_transition(from_label as u32, to_label as u32);
                if weight.abs() > 1e-10 {
                    self.features.push(FeatureFunction::new_transition(
                        id,
                        from_label as u32,
                        to_label as u32,
                        weight,
                    ));
                    id += 1;
                }
            }
        }
    }

    /// Apply L2 regularization decay to all weights.
    pub fn apply_l2_decay(&mut self, factor: f64) {
        for weight in self.state_weights.values_mut() {
            *weight *= factor;
        }
        for weight in &mut self.transition_weights {
            *weight *= factor;
        }
    }

    /// Apply L1 regularization (soft thresholding).
    pub fn apply_l1_penalty(&mut self, penalty: f64) {
        // Soft thresholding for state weights
        self.state_weights.retain(|_, weight| {
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

        // Soft thresholding for transition weights
        for weight in &mut self.transition_weights {
            if *weight > penalty {
                *weight -= penalty;
            } else if *weight < -penalty {
                *weight += penalty;
            } else {
                *weight = 0.0;
            }
        }
    }

    /// Get the squared L2 norm of all weights.
    pub fn l2_norm_squared(&self) -> f64 {
        let state_norm: f64 = self.state_weights.values().map(|w| w * w).sum();
        let trans_norm: f64 = self.transition_weights.iter().map(|w| w * w).sum();
        state_norm + trans_norm
    }

    /// Get the L1 norm of all weights.
    pub fn l1_norm(&self) -> f64 {
        let state_norm: f64 = self.state_weights.values().map(|w| w.abs()).sum();
        let trans_norm: f64 = self.transition_weights.iter().map(|w| w.abs()).sum();
        state_norm + trans_norm
    }

    /// Convert attribute strings to IDs, creating new IDs if necessary.
    pub fn attrs_to_ids(&mut self, attrs: &[String]) -> Vec<u32> {
        attrs
            .iter()
            .map(|a| self.attributes.get_or_insert(a))
            .collect()
    }

    /// Convert attribute strings to IDs (read-only, returns None for unknown).
    pub fn attrs_to_ids_readonly(&self, attrs: &[String]) -> Vec<u32> {
        attrs
            .iter()
            .filter_map(|a| self.attributes.get(a))
            .collect()
    }

    /// Convert label string to ID, creating new ID if necessary.
    pub fn label_to_id(&mut self, label: &str) -> u32 {
        let id = self.labels.get_or_insert(label);
        self.num_labels = self.labels.len();
        if self.transition_weights.len() != self.num_labels * self.num_labels {
            self.initialize_transition_weights();
        }
        id
    }

    /// Convert label string to ID (read-only).
    pub fn label_to_id_readonly(&self, label: &str) -> Option<u32> {
        self.labels.get(label)
    }

    /// Convert label ID to string.
    pub fn id_to_label(&self, id: u32) -> Option<&str> {
        self.labels.get_label(id)
    }

    /// Iterate over state weights.
    pub fn state_weights_iter(&self) -> impl Iterator<Item = (&(u32, u32), &f64)> {
        self.state_weights.iter()
    }
}

impl Default for CRFModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_model() {
        let model = CRFModel::new();
        assert_eq!(model.num_labels, 0);
        assert_eq!(model.num_attributes, 0);
    }

    #[test]
    fn test_model_with_labels() {
        let model = CRFModel::with_labels(vec![
            "B-PER".to_string(),
            "I-PER".to_string(),
            "O".to_string(),
        ]);
        assert_eq!(model.num_labels, 3);
        assert_eq!(model.labels.get("B-PER"), Some(0));
        assert_eq!(model.labels.get("I-PER"), Some(1));
        assert_eq!(model.labels.get("O"), Some(2));
    }

    #[test]
    fn test_transition_weights() {
        let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

        model.set_transition(0, 1, 0.5);
        model.set_transition(1, 0, -0.3);

        assert_eq!(model.get_transition(0, 1), 0.5);
        assert_eq!(model.get_transition(1, 0), -0.3);
        assert_eq!(model.get_transition(0, 0), 0.0); // Default

        model.add_transition(0, 1, 0.5);
        assert_eq!(model.get_transition(0, 1), 1.0);
    }

    #[test]
    fn test_state_weights() {
        let mut model = CRFModel::new();

        model.set_state_weight(0, 0, 0.5);
        model.set_state_weight(0, 1, -0.3);
        model.set_state_weight(1, 0, 0.2);

        assert_eq!(model.get_state_weight(0, 0), 0.5);
        assert_eq!(model.get_state_weight(0, 1), -0.3);
        assert_eq!(model.get_state_weight(1, 0), 0.2);
        assert_eq!(model.get_state_weight(1, 1), 0.0); // Default

        model.add_state_weight(0, 0, 0.5);
        assert_eq!(model.get_state_weight(0, 0), 1.0);
    }

    #[test]
    fn test_emission_scores() {
        let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

        model.set_state_weight(0, 0, 1.0); // attr 0 -> label A
        model.set_state_weight(0, 1, 0.5); // attr 0 -> label B
        model.set_state_weight(1, 0, 0.3); // attr 1 -> label A

        let attr_ids = vec![0, 1];
        let scores = model.emission_scores(&attr_ids);

        assert_eq!(scores[0], 1.3); // 1.0 + 0.3
        assert_eq!(scores[1], 0.5); // 0.5 + 0.0
    }

    #[test]
    fn test_l2_regularization() {
        let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

        model.set_state_weight(0, 0, 1.0);
        model.set_transition(0, 1, 2.0);

        model.apply_l2_decay(0.5);

        assert_eq!(model.get_state_weight(0, 0), 0.5);
        assert_eq!(model.get_transition(0, 1), 1.0);
    }

    #[test]
    fn test_l1_regularization() {
        let mut model = CRFModel::with_labels(vec!["A".to_string(), "B".to_string()]);

        model.set_state_weight(0, 0, 1.0);
        model.set_state_weight(0, 1, 0.1); // Will be zeroed
        model.set_transition(0, 1, 2.0);

        model.apply_l1_penalty(0.2);

        assert_eq!(model.get_state_weight(0, 0), 0.8);
        assert_eq!(model.get_state_weight(0, 1), 0.0); // Removed
        assert_eq!(model.get_transition(0, 1), 1.8);
    }
}
