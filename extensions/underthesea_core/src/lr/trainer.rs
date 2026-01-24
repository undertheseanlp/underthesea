//! Logistic Regression Trainer with SGD optimization.
//!
//! This module provides training functionality for LR models:
//! - Mini-batch SGD optimization
//! - L1 and L2 regularization
//! - Cross-entropy loss with softmax

use super::model::LRModel;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// A training instance consisting of features and a label.
#[derive(Debug, Clone)]
pub struct TrainingInstance {
    /// Feature strings for this instance
    pub features: Vec<String>,

    /// Class label for this instance
    pub label: String,
}

impl TrainingInstance {
    /// Create a new training instance.
    pub fn new(features: Vec<String>, label: String) -> Self {
        Self { features, label }
    }
}

/// Configuration for the LR trainer.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// L1 regularization penalty (lasso)
    pub l1_penalty: f64,

    /// L2 regularization penalty (ridge)
    pub l2_penalty: f64,

    /// Learning rate for SGD
    pub learning_rate: f64,

    /// Maximum number of training epochs
    pub max_epochs: usize,

    /// Mini-batch size (1 = pure SGD, larger = mini-batch SGD)
    pub batch_size: usize,

    /// Convergence tolerance for early stopping
    pub tol: f64,

    /// Verbosity level (0=quiet, 1=progress, 2=detailed)
    pub verbose: u8,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            l1_penalty: 0.0,
            l2_penalty: 0.01,
            learning_rate: 0.1,
            max_epochs: 100,
            batch_size: 1,
            tol: 1e-4,
            verbose: 1,
        }
    }
}

/// Logistic Regression trainer using SGD.
pub struct LRTrainer {
    config: TrainerConfig,
    model: LRModel,
}

impl LRTrainer {
    /// Create a new trainer with default configuration.
    pub fn new() -> Self {
        Self {
            config: TrainerConfig::default(),
            model: LRModel::new(),
        }
    }

    /// Create a new trainer with custom configuration.
    pub fn with_config(config: TrainerConfig) -> Self {
        Self {
            config,
            model: LRModel::new(),
        }
    }

    /// Set L1 regularization penalty.
    pub fn set_l1_penalty(&mut self, penalty: f64) {
        self.config.l1_penalty = penalty;
    }

    /// Set L2 regularization penalty.
    pub fn set_l2_penalty(&mut self, penalty: f64) {
        self.config.l2_penalty = penalty;
    }

    /// Set learning rate.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Set maximum epochs.
    pub fn set_max_epochs(&mut self, epochs: usize) {
        self.config.max_epochs = epochs;
    }

    /// Set batch size.
    pub fn set_batch_size(&mut self, size: usize) {
        self.config.batch_size = size.max(1);
    }

    /// Set convergence tolerance.
    pub fn set_tol(&mut self, tol: f64) {
        self.config.tol = tol;
    }

    /// Set verbosity level.
    pub fn set_verbose(&mut self, verbose: u8) {
        self.config.verbose = verbose;
    }

    /// Get the current model.
    pub fn get_model(&self) -> &LRModel {
        &self.model
    }

    /// Train the model on the given data.
    pub fn train(&mut self, data: &[TrainingInstance]) -> LRModel {
        if data.is_empty() {
            return self.model.clone();
        }

        // Build vocabulary and class labels
        self.build_indices(data);

        // Convert data to internal representation
        let instances: Vec<(Vec<u32>, u32)> = data
            .iter()
            .map(|inst| {
                let feature_ids = self.model.features_to_ids_readonly(&inst.features);
                let class_id = self.model.class_to_id_readonly(&inst.label).unwrap_or(0);
                (feature_ids, class_id)
            })
            .collect();

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..instances.len()).collect();
        let mut rng = thread_rng();

        let mut prev_loss = f64::INFINITY;

        // Training loop
        for epoch in 0..self.config.max_epochs {
            // Shuffle data
            indices.shuffle(&mut rng);

            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Process mini-batches
            for batch_start in (0..instances.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(instances.len());
                let batch_indices = &indices[batch_start..batch_end];

                let batch_loss = self.train_batch(&instances, batch_indices);
                epoch_loss += batch_loss;
                num_batches += 1;
            }

            // Apply regularization at the end of each epoch
            if self.config.l2_penalty > 0.0 {
                let decay = 1.0 - self.config.learning_rate * self.config.l2_penalty;
                self.model.apply_l2_decay(decay.max(0.0));
            }

            if self.config.l1_penalty > 0.0 {
                let penalty = self.config.learning_rate * self.config.l1_penalty;
                self.model.apply_l1_penalty(penalty);
            }

            let avg_loss = epoch_loss / num_batches as f64;

            if self.config.verbose >= 1 {
                eprintln!(
                    "Epoch {}/{}: loss={:.6}",
                    epoch + 1,
                    self.config.max_epochs,
                    avg_loss
                );
            }

            // Check for convergence
            if (prev_loss - avg_loss).abs() < self.config.tol {
                if self.config.verbose >= 1 {
                    eprintln!("Converged at epoch {}", epoch + 1);
                }
                break;
            }
            prev_loss = avg_loss;
        }

        self.model.clone()
    }

    /// Build feature and class indices from training data.
    fn build_indices(&mut self, data: &[TrainingInstance]) {
        self.model = LRModel::new();

        // First pass: collect all classes
        for inst in data {
            self.model.class_to_id(&inst.label);
        }

        // Second pass: collect all features
        for inst in data {
            self.model.features_to_ids(&inst.features);
        }

        if self.config.verbose >= 2 {
            eprintln!(
                "Built indices: {} classes, {} features",
                self.model.num_classes, self.model.num_features
            );
        }
    }

    /// Train on a mini-batch and return the batch loss.
    fn train_batch(&mut self, instances: &[(Vec<u32>, u32)], batch_indices: &[usize]) -> f64 {
        let num_classes = self.model.num_classes;
        let batch_size = batch_indices.len() as f64;

        // Accumulate gradients
        let mut weight_grads: hashbrown::HashMap<(u32, u32), f64> = hashbrown::HashMap::new();
        let mut bias_grads: Vec<f64> = vec![0.0; num_classes];
        let mut total_loss = 0.0;

        for &idx in batch_indices {
            let (feature_ids, true_class) = &instances[idx];

            // Forward pass
            let logits = self.model.compute_logits(feature_ids);
            let probs = LRModel::softmax(&logits);

            // Compute loss (negative log likelihood)
            let prob_true = probs[*true_class as usize].max(1e-15);
            total_loss -= prob_true.ln();

            // Backward pass: compute gradients
            // Gradient of softmax cross-entropy: prob - one_hot
            for class_id in 0..num_classes {
                let target = if class_id == *true_class as usize {
                    1.0
                } else {
                    0.0
                };
                let grad = probs[class_id] - target;

                // Gradient for bias
                bias_grads[class_id] += grad;

                // Gradient for weights
                for &feature_id in feature_ids {
                    *weight_grads
                        .entry((feature_id, class_id as u32))
                        .or_insert(0.0) += grad;
                }
            }
        }

        // Apply gradients (SGD update)
        let lr = self.config.learning_rate / batch_size;

        for ((feature_id, class_id), grad) in weight_grads {
            self.model.add_weight(feature_id, class_id, -lr * grad);
        }

        for (class_id, grad) in bias_grads.iter().enumerate() {
            self.model.add_bias(class_id as u32, -lr * grad);
        }

        total_loss / batch_size
    }
}

impl Default for LRTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_data() -> Vec<TrainingInstance> {
        vec![
            TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
            TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
            TrainingInstance::new(vec!["f1=a".to_string()], "A".to_string()),
            TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
            TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
            TrainingInstance::new(vec!["f1=b".to_string()], "B".to_string()),
        ]
    }

    #[test]
    fn test_trainer_new() {
        let trainer = LRTrainer::new();
        assert_eq!(trainer.config.max_epochs, 100);
        assert_eq!(trainer.config.learning_rate, 0.1);
    }

    #[test]
    fn test_trainer_with_config() {
        let config = TrainerConfig {
            max_epochs: 50,
            learning_rate: 0.5,
            ..Default::default()
        };
        let trainer = LRTrainer::with_config(config);
        assert_eq!(trainer.config.max_epochs, 50);
        assert_eq!(trainer.config.learning_rate, 0.5);
    }

    #[test]
    fn test_training_simple() {
        let data = create_simple_data();
        let config = TrainerConfig {
            max_epochs: 50,
            learning_rate: 1.0,
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&data);

        assert_eq!(model.num_classes, 2);
        assert_eq!(model.num_features, 2);

        // The model should have learned to classify correctly
        let feat_a_ids = model.features_to_ids_readonly(&["f1=a".to_string()]);
        let feat_b_ids = model.features_to_ids_readonly(&["f1=b".to_string()]);

        let logits_a = model.compute_logits(&feat_a_ids);
        let logits_b = model.compute_logits(&feat_b_ids);

        let class_a_id = model.class_to_id_readonly("A").unwrap() as usize;
        let class_b_id = model.class_to_id_readonly("B").unwrap() as usize;

        // Feature "f1=a" should predict class A
        assert!(
            logits_a[class_a_id] > logits_a[class_b_id],
            "f1=a should predict A"
        );

        // Feature "f1=b" should predict class B
        assert!(
            logits_b[class_b_id] > logits_b[class_a_id],
            "f1=b should predict B"
        );
    }

    #[test]
    fn test_training_multiclass() {
        let data = vec![
            TrainingInstance::new(vec!["color=red".to_string()], "apple".to_string()),
            TrainingInstance::new(vec!["color=yellow".to_string()], "banana".to_string()),
            TrainingInstance::new(vec!["color=orange".to_string()], "orange".to_string()),
            TrainingInstance::new(vec!["color=red".to_string()], "apple".to_string()),
            TrainingInstance::new(vec!["color=yellow".to_string()], "banana".to_string()),
            TrainingInstance::new(vec!["color=orange".to_string()], "orange".to_string()),
        ];

        let config = TrainerConfig {
            max_epochs: 100,
            learning_rate: 1.0,
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&data);

        assert_eq!(model.num_classes, 3);
    }

    #[test]
    fn test_training_with_l2() {
        let data = create_simple_data();
        let config = TrainerConfig {
            max_epochs: 10,
            learning_rate: 0.5,
            l2_penalty: 0.1,
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&data);

        // L2 regularization should keep weights smaller
        assert!(model.l2_norm_squared() < 100.0);
    }

    #[test]
    fn test_training_with_l1() {
        let data = create_simple_data();
        let config = TrainerConfig {
            max_epochs: 10,
            learning_rate: 0.5,
            l1_penalty: 0.1,
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&data);

        // L1 regularization encourages sparsity
        assert!(model.num_weights() <= model.num_features * model.num_classes);
    }

    #[test]
    fn test_training_empty_data() {
        let config = TrainerConfig {
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&[]);

        assert_eq!(model.num_classes, 0);
        assert_eq!(model.num_features, 0);
    }

    #[test]
    fn test_mini_batch() {
        let data = create_simple_data();
        let config = TrainerConfig {
            max_epochs: 20,
            learning_rate: 0.5,
            batch_size: 2,
            verbose: 0,
            ..Default::default()
        };
        let mut trainer = LRTrainer::with_config(config);
        let model = trainer.train(&data);

        assert_eq!(model.num_classes, 2);
    }
}
