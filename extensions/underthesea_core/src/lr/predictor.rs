//! Logistic Regression Classifier for prediction.
//!
//! This module provides prediction functionality for trained LR models:
//! - predict: Get the most likely class label
//! - predict_proba: Get probability distribution over all classes
//! - predict_top_k: Get top-k most likely classes with probabilities

use super::model::LRModel;
use super::serialization::{LRFormat, LRModelLoader};

/// A classifier that uses a trained LR model for prediction.
#[derive(Debug, Clone)]
pub struct LRClassifier {
    model: LRModel,
}

impl LRClassifier {
    /// Create a new classifier with an empty model.
    pub fn new() -> Self {
        Self {
            model: LRModel::new(),
        }
    }

    /// Create a classifier from an existing model.
    pub fn from_model(model: LRModel) -> Self {
        Self { model }
    }

    /// Load a model from a file.
    pub fn load(path: String) -> Result<Self, String> {
        let loader = LRModelLoader::new();
        let model = loader.load(path, LRFormat::Auto)?;
        Ok(Self { model })
    }

    /// Get the underlying model.
    pub fn model(&self) -> &LRModel {
        &self.model
    }

    /// Predict the most likely class for the given features.
    ///
    /// Returns the class label with the highest probability.
    pub fn predict(&self, features: &[String]) -> String {
        let feature_ids = self.model.features_to_ids_readonly(features);
        let logits = self.model.compute_logits(&feature_ids);

        if logits.is_empty() {
            return String::new();
        }

        // Find argmax
        let (best_class_id, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        self.model
            .id_to_class(best_class_id as u32)
            .unwrap_or("")
            .to_string()
    }

    /// Predict with probability for the most likely class.
    ///
    /// Returns (class_label, probability).
    pub fn predict_with_prob(&self, features: &[String]) -> (String, f64) {
        let feature_ids = self.model.features_to_ids_readonly(features);
        let logits = self.model.compute_logits(&feature_ids);

        if logits.is_empty() {
            return (String::new(), 0.0);
        }

        let probs = LRModel::softmax(&logits);

        // Find argmax
        let (best_class_id, &best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let class_label = self
            .model
            .id_to_class(best_class_id as u32)
            .unwrap_or("")
            .to_string();

        (class_label, best_prob)
    }

    /// Get probability distribution over all classes.
    ///
    /// Returns a list of (class_label, probability) tuples, sorted by probability descending.
    pub fn predict_proba(&self, features: &[String]) -> Vec<(String, f64)> {
        let feature_ids = self.model.features_to_ids_readonly(features);
        let logits = self.model.compute_logits(&feature_ids);

        if logits.is_empty() {
            return Vec::new();
        }

        let probs = LRModel::softmax(&logits);

        let mut result: Vec<(String, f64)> = probs
            .iter()
            .enumerate()
            .map(|(class_id, &prob)| {
                let label = self
                    .model
                    .id_to_class(class_id as u32)
                    .unwrap_or("")
                    .to_string();
                (label, prob)
            })
            .collect();

        // Sort by probability descending
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        result
    }

    /// Get top-k most likely classes with probabilities.
    ///
    /// Returns a list of (class_label, probability) tuples for the top-k classes.
    pub fn predict_top_k(&self, features: &[String], k: usize) -> Vec<(String, f64)> {
        let mut proba = self.predict_proba(features);
        proba.truncate(k);
        proba
    }

    /// Get the number of classes.
    pub fn num_classes(&self) -> usize {
        self.model.num_classes
    }

    /// Get all class labels.
    pub fn classes(&self) -> Vec<String> {
        self.model.get_classes()
    }
}

impl Default for LRClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> LRModel {
        let mut model = LRModel::with_classes(vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]);

        // Add some features
        let f1 = model.features.get_or_insert("word=good");
        let f2 = model.features.get_or_insert("word=bad");
        let f3 = model.features.get_or_insert("word=okay");
        model.num_features = model.features.len();

        // Set weights that make "good" -> positive, "bad" -> negative, "okay" -> neutral
        model.set_weight(f1, 0, 2.0); // good -> positive
        model.set_weight(f1, 1, -1.0); // good -> negative (negative weight)
        model.set_weight(f2, 1, 2.0); // bad -> negative
        model.set_weight(f2, 0, -1.0); // bad -> positive (negative weight)
        model.set_weight(f3, 2, 2.0); // okay -> neutral

        model
    }

    #[test]
    fn test_classifier_new() {
        let classifier = LRClassifier::new();
        assert_eq!(classifier.num_classes(), 0);
    }

    #[test]
    fn test_classifier_from_model() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);
        assert_eq!(classifier.num_classes(), 3);
    }

    #[test]
    fn test_predict() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        // "good" should predict "positive"
        let pred = classifier.predict(&["word=good".to_string()]);
        assert_eq!(pred, "positive");

        // "bad" should predict "negative"
        let pred = classifier.predict(&["word=bad".to_string()]);
        assert_eq!(pred, "negative");

        // "okay" should predict "neutral"
        let pred = classifier.predict(&["word=okay".to_string()]);
        assert_eq!(pred, "neutral");
    }

    #[test]
    fn test_predict_with_prob() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        let (label, prob) = classifier.predict_with_prob(&["word=good".to_string()]);
        assert_eq!(label, "positive");
        assert!(prob > 0.5, "Probability should be > 0.5, got {}", prob);
    }

    #[test]
    fn test_predict_proba() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        let proba = classifier.predict_proba(&["word=good".to_string()]);

        // Should have 3 classes
        assert_eq!(proba.len(), 3);

        // Should be sorted by probability descending
        for i in 0..proba.len() - 1 {
            assert!(proba[i].1 >= proba[i + 1].1);
        }

        // Probabilities should sum to ~1
        let sum: f64 = proba.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // "positive" should have highest probability
        assert_eq!(proba[0].0, "positive");
    }

    #[test]
    fn test_predict_top_k() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        let top2 = classifier.predict_top_k(&["word=good".to_string()], 2);
        assert_eq!(top2.len(), 2);

        // First should be "positive"
        assert_eq!(top2[0].0, "positive");
    }

    #[test]
    fn test_predict_unknown_features() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        // Unknown features should still work (just use biases)
        let pred = classifier.predict(&["word=unknown".to_string()]);
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_get_classes() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        let classes = classifier.classes();
        assert_eq!(classes.len(), 3);
        assert!(classes.contains(&"positive".to_string()));
        assert!(classes.contains(&"negative".to_string()));
        assert!(classes.contains(&"neutral".to_string()));
    }

    #[test]
    fn test_empty_features() {
        let model = create_test_model();
        let classifier = LRClassifier::from_model(model);

        // Empty features should use only biases
        let pred = classifier.predict(&[]);
        assert!(!pred.is_empty());

        let proba = classifier.predict_proba(&[]);
        assert_eq!(proba.len(), 3);
    }
}
