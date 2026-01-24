//! CRF Training algorithms.
//!
//! This module provides training algorithms for CRF models:
//! - Negative Log-Likelihood (NLL) with L-BFGS optimization
//! - Structured Perceptron (online learning)
//!
//! Both methods support L1 and L2 regularization.

use super::model::CRFModel;
use super::tagger::CRFTagger;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Loss function types for CRF training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    /// Negative log-likelihood with regularization
    NegativeLogLikelihood { l1_penalty: f64, l2_penalty: f64 },

    /// Structured Perceptron (online learning)
    StructuredPerceptron { learning_rate: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::NegativeLogLikelihood {
            l1_penalty: 0.0,
            l2_penalty: 0.01,
        }
    }
}

/// Configuration for CRF training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Loss function to use
    pub loss_function: LossFunction,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence threshold (minimum improvement)
    pub epsilon: f64,

    /// Number of past iterations to average (for averaged perceptron)
    pub averaging: bool,

    /// Verbosity level (0=quiet, 1=progress, 2=detailed)
    pub verbose: u8,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            loss_function: LossFunction::default(),
            max_iterations: 100,
            epsilon: 1e-5,
            averaging: true,
            verbose: 1,
        }
    }
}

/// A training instance (sequence of features and labels).
#[derive(Debug, Clone)]
pub struct TrainingInstance {
    /// Feature vectors for each token
    pub features: Vec<Vec<String>>,

    /// Gold labels for each token
    pub labels: Vec<String>,
}

impl TrainingInstance {
    /// Create a new training instance.
    pub fn new(features: Vec<Vec<String>>, labels: Vec<String>) -> Self {
        Self { features, labels }
    }
}

/// CRF Trainer for learning model parameters.
pub struct CRFTrainer {
    /// Training configuration
    config: TrainerConfig,

    /// The model being trained
    model: CRFModel,

    /// Tagger for Viterbi decoding during training
    tagger: CRFTagger,

    /// Averaged weights (for averaged perceptron)
    avg_state_weights: HashMap<(u32, u32), f64>,
    avg_transition_weights: Vec<f64>,

    /// Update counter (for averaging)
    update_count: usize,
}

impl CRFTrainer {
    /// Create a new trainer with default configuration.
    pub fn new() -> Self {
        Self::with_config(TrainerConfig::default())
    }

    /// Create a new trainer with custom configuration.
    pub fn with_config(config: TrainerConfig) -> Self {
        Self {
            config,
            model: CRFModel::new(),
            tagger: CRFTagger::new(),
            avg_state_weights: HashMap::new(),
            avg_transition_weights: Vec::new(),
            update_count: 0,
        }
    }

    /// Set the loss function.
    pub fn set_loss_function(&mut self, loss: LossFunction) {
        self.config.loss_function = loss;
    }

    /// Set maximum iterations.
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.config.max_iterations = max_iter;
    }

    /// Set L1 penalty (only for NLL).
    pub fn set_l1_penalty(&mut self, penalty: f64) {
        if let LossFunction::NegativeLogLikelihood { l2_penalty, .. } = self.config.loss_function {
            self.config.loss_function = LossFunction::NegativeLogLikelihood {
                l1_penalty: penalty,
                l2_penalty,
            };
        }
    }

    /// Set L2 penalty (only for NLL).
    pub fn set_l2_penalty(&mut self, penalty: f64) {
        if let LossFunction::NegativeLogLikelihood { l1_penalty, .. } = self.config.loss_function {
            self.config.loss_function = LossFunction::NegativeLogLikelihood {
                l1_penalty,
                l2_penalty: penalty,
            };
        }
    }

    /// Train the model on the given data.
    ///
    /// # Arguments
    /// * `data` - Training instances (sequences of features and labels)
    ///
    /// # Returns
    /// * Trained CRFModel
    pub fn train(&mut self, data: &[TrainingInstance]) -> CRFModel {
        // Initialize model from data
        self.initialize_from_data(data);

        // Train based on loss function
        match self.config.loss_function.clone() {
            LossFunction::NegativeLogLikelihood {
                l1_penalty,
                l2_penalty,
            } => {
                self.train_nll(data, l1_penalty, l2_penalty);
            }
            LossFunction::StructuredPerceptron { learning_rate } => {
                self.train_perceptron(data, learning_rate);
            }
        }

        self.model.clone()
    }

    /// Initialize model from training data (collect labels and attributes).
    fn initialize_from_data(&mut self, data: &[TrainingInstance]) {
        // Collect all labels
        for instance in data {
            for label in &instance.labels {
                self.model.label_to_id(label);
            }
        }

        // Collect all attributes
        for instance in data {
            for token_features in &instance.features {
                for attr in token_features {
                    self.model.attributes.get_or_insert(attr);
                }
            }
        }

        self.model.num_attributes = self.model.attributes.len();
        self.tagger = CRFTagger::from_model(self.model.clone());

        // Initialize averaging weights
        if self.config.averaging {
            self.avg_transition_weights = vec![0.0; self.model.num_labels * self.model.num_labels];
        }

        if self.config.verbose > 0 {
            eprintln!(
                "Initialized model with {} labels and {} attributes",
                self.model.num_labels,
                self.model.num_attributes
            );
        }
    }

    /// Train using Negative Log-Likelihood with SGD.
    fn train_nll(&mut self, data: &[TrainingInstance], l1_penalty: f64, l2_penalty: f64) {
        let learning_rate = 0.1;
        let mut prev_loss = f64::MAX;

        for iter in 0..self.config.max_iterations {
            let mut total_loss = 0.0;

            for instance in data {
                let loss = self.nll_update(instance, learning_rate, l2_penalty);
                total_loss += loss;
            }

            // Apply L1 regularization at the end of each iteration
            if l1_penalty > 0.0 {
                self.model.apply_l1_penalty(l1_penalty * learning_rate);
            }

            // Add regularization to loss
            total_loss += 0.5 * l2_penalty * self.model.l2_norm_squared();
            total_loss += l1_penalty * self.model.l1_norm();

            if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                eprintln!("Iteration {}: loss = {:.4}", iter + 1, total_loss);
            }

            // Check convergence
            if (prev_loss - total_loss).abs() < self.config.epsilon {
                if self.config.verbose > 0 {
                    eprintln!("Converged at iteration {}", iter + 1);
                }
                break;
            }
            prev_loss = total_loss;

            // Update tagger with new model
            self.tagger = CRFTagger::from_model(self.model.clone());
        }
    }

    /// Compute NLL gradient and update weights for a single instance.
    fn nll_update(&mut self, instance: &TrainingInstance, learning_rate: f64, l2_penalty: f64) -> f64 {
        let n = instance.features.len();
        if n == 0 {
            return 0.0;
        }

        // Convert to IDs
        let attr_ids: Vec<Vec<u32>> = instance
            .features
            .iter()
            .map(|f| self.model.attrs_to_ids_readonly(f))
            .collect();

        let gold_labels: Vec<u32> = instance
            .labels
            .iter()
            .filter_map(|l| self.model.label_to_id_readonly(l))
            .collect();

        if gold_labels.len() != n {
            return 0.0;
        }

        // Forward-backward
        let (alpha, log_z) = self.tagger.forward(&attr_ids);
        let beta = self.tagger.backward(&attr_ids);

        // Compute gold score
        let gold_score = self.tagger.sequence_score(&attr_ids, &gold_labels);

        // NLL = log_z - gold_score
        let loss = log_z - gold_score;

        // Compute gradients and update weights
        let num_labels = self.model.num_labels;

        // State feature gradients
        for t in 0..n {
            let gold_label = gold_labels[t] as usize;

            // Increase weight for gold features
            for &attr_id in &attr_ids[t] {
                self.model
                    .add_state_weight(attr_id, gold_label as u32, learning_rate);
            }

            // Decrease weight for expected features
            for y in 0..num_labels {
                let prob = (alpha[t][y] + beta[t][y] - log_z).exp();
                for &attr_id in &attr_ids[t] {
                    self.model
                        .add_state_weight(attr_id, y as u32, -learning_rate * prob);
                }
            }
        }

        // Transition feature gradients
        for t in 1..n {
            let gold_prev = gold_labels[t - 1];
            let gold_curr = gold_labels[t];

            // Increase weight for gold transition
            self.model.add_transition(gold_prev, gold_curr, learning_rate);

            // Decrease weight for expected transitions
            let emission_t = self.model.emission_scores(&attr_ids[t]);
            for y_prev in 0..num_labels {
                for y_curr in 0..num_labels {
                    let trans = self.model.get_transition(y_prev as u32, y_curr as u32);
                    let log_prob = alpha[t - 1][y_prev] + trans + emission_t[y_curr] + beta[t][y_curr] - log_z;
                    let prob = log_prob.exp();
                    self.model
                        .add_transition(y_prev as u32, y_curr as u32, -learning_rate * prob);
                }
            }
        }

        // Apply L2 regularization (weight decay)
        if l2_penalty > 0.0 {
            self.model.apply_l2_decay(1.0 - learning_rate * l2_penalty);
        }

        loss
    }

    /// Train using Structured Perceptron algorithm.
    fn train_perceptron(&mut self, data: &[TrainingInstance], learning_rate: f64) {
        let mut num_errors = 0;
        let mut prev_errors = usize::MAX;

        for iter in 0..self.config.max_iterations {
            num_errors = 0;

            for instance in data {
                if self.perceptron_update(instance, learning_rate) {
                    num_errors += 1;
                }
                self.update_count += 1;

                // Update running average
                if self.config.averaging {
                    self.update_average();
                }
            }

            if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                eprintln!("Iteration {}: {} errors", iter + 1, num_errors);
            }

            // Check convergence (no errors or no improvement)
            if num_errors == 0 {
                if self.config.verbose > 0 {
                    eprintln!("Converged at iteration {} (no errors)", iter + 1);
                }
                break;
            }
            if num_errors >= prev_errors && iter > 10 {
                if self.config.verbose > 0 {
                    eprintln!("No improvement at iteration {}", iter + 1);
                }
                break;
            }
            prev_errors = num_errors;

            // Update tagger with new model
            self.tagger = CRFTagger::from_model(self.model.clone());
        }

        // Apply averaging
        if self.config.averaging && self.update_count > 0 {
            self.apply_average();
        }
    }

    /// Perform a single perceptron update.
    /// Returns true if an error was made.
    fn perceptron_update(&mut self, instance: &TrainingInstance, learning_rate: f64) -> bool {
        let n = instance.features.len();
        if n == 0 {
            return false;
        }

        // Get predicted labels
        let predicted = self.tagger.tag(&instance.features);

        // Check if prediction matches gold
        if predicted == instance.labels {
            return false;
        }

        // Convert to IDs
        let gold_labels: Vec<u32> = instance
            .labels
            .iter()
            .filter_map(|l| self.model.label_to_id_readonly(l))
            .collect();

        let pred_labels: Vec<u32> = predicted
            .iter()
            .filter_map(|l| self.model.label_to_id_readonly(l))
            .collect();

        if gold_labels.len() != n || pred_labels.len() != n {
            return false;
        }

        // Update weights: increase for gold, decrease for predicted
        for t in 0..n {
            let gold_y = gold_labels[t];
            let pred_y = pred_labels[t];

            if gold_y != pred_y {
                // Update state features
                for feat in &instance.features[t] {
                    if let Some(attr_id) = self.model.attributes.get(feat) {
                        self.model.add_state_weight(attr_id, gold_y, learning_rate);
                        self.model.add_state_weight(attr_id, pred_y, -learning_rate);
                    }
                }
            }
        }

        // Update transition features
        for t in 1..n {
            let gold_prev = gold_labels[t - 1];
            let gold_curr = gold_labels[t];
            let pred_prev = pred_labels[t - 1];
            let pred_curr = pred_labels[t];

            if gold_prev != pred_prev || gold_curr != pred_curr {
                self.model.add_transition(gold_prev, gold_curr, learning_rate);
                self.model.add_transition(pred_prev, pred_curr, -learning_rate);
            }
        }

        true
    }

    /// Update running average for averaged perceptron.
    fn update_average(&mut self) {
        // This is a simplified averaging that stores the sum of all weights
        // The final average is computed by dividing by update_count

        // Update averaged state weights
        for (&key, &weight) in self.model.state_weights_iter() {
            *self.avg_state_weights.entry(key).or_insert(0.0) += weight;
        }

        // Update averaged transition weights
        let n = self.model.num_labels;
        if self.avg_transition_weights.len() != n * n {
            self.avg_transition_weights = vec![0.0; n * n];
        }
        for (i, &w) in self.model.transition_weights().iter().enumerate() {
            self.avg_transition_weights[i] += w;
        }
    }

    /// Apply averaged weights to the model.
    fn apply_average(&mut self) {
        if self.update_count == 0 {
            return;
        }

        let count = self.update_count as f64;

        // Apply averaged state weights
        for (&key, &sum) in &self.avg_state_weights {
            self.model.set_state_weight(key.0, key.1, sum / count);
        }

        // Apply averaged transition weights
        let n = self.model.num_labels;
        for from_label in 0..n {
            for to_label in 0..n {
                let idx = from_label * n + to_label;
                if idx < self.avg_transition_weights.len() {
                    let avg = self.avg_transition_weights[idx] / count;
                    self.model.set_transition(from_label as u32, to_label as u32, avg);
                }
            }
        }
    }

    /// Get the current model (during or after training).
    pub fn get_model(&self) -> &CRFModel {
        &self.model
    }
}

impl Default for CRFTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_training_data() -> Vec<TrainingInstance> {
        vec![
            TrainingInstance::new(
                vec![
                    vec!["word=John".to_string(), "cap=True".to_string()],
                    vec!["word=Smith".to_string(), "cap=True".to_string()],
                    vec!["word=is".to_string(), "cap=False".to_string()],
                    vec!["word=here".to_string(), "cap=False".to_string()],
                ],
                vec![
                    "B-PER".to_string(),
                    "I-PER".to_string(),
                    "O".to_string(),
                    "O".to_string(),
                ],
            ),
            TrainingInstance::new(
                vec![
                    vec!["word=Paris".to_string(), "cap=True".to_string()],
                    vec!["word=is".to_string(), "cap=False".to_string()],
                    vec!["word=beautiful".to_string(), "cap=False".to_string()],
                ],
                vec!["B-LOC".to_string(), "O".to_string(), "O".to_string()],
            ),
        ]
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = CRFTrainer::new();
        assert_eq!(trainer.config.max_iterations, 100);
    }

    #[test]
    fn test_trainer_with_config() {
        let config = TrainerConfig {
            max_iterations: 50,
            loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.1 },
            ..Default::default()
        };
        let trainer = CRFTrainer::with_config(config);
        assert_eq!(trainer.config.max_iterations, 50);
    }

    #[test]
    fn test_initialize_from_data() {
        let mut trainer = CRFTrainer::new();
        trainer.config.verbose = 0;
        let data = create_training_data();
        trainer.initialize_from_data(&data);

        assert_eq!(trainer.model.num_labels, 4); // B-PER, I-PER, O, B-LOC
        assert!(trainer.model.num_attributes > 0);
    }

    #[test]
    fn test_train_perceptron() {
        let mut trainer = CRFTrainer::with_config(TrainerConfig {
            loss_function: LossFunction::StructuredPerceptron { learning_rate: 0.1 },
            max_iterations: 10,
            verbose: 0,
            ..Default::default()
        });

        let data = create_training_data();
        let model = trainer.train(&data);

        assert_eq!(model.num_labels, 4);
        assert!(model.num_state_features() > 0);
    }

    #[test]
    fn test_train_nll() {
        let mut trainer = CRFTrainer::with_config(TrainerConfig {
            loss_function: LossFunction::NegativeLogLikelihood {
                l1_penalty: 0.0,
                l2_penalty: 0.01,
            },
            max_iterations: 5,
            verbose: 0,
            ..Default::default()
        });

        let data = create_training_data();
        let model = trainer.train(&data);

        assert_eq!(model.num_labels, 4);
    }

    #[test]
    fn test_training_instance() {
        let instance = TrainingInstance::new(
            vec![vec!["word=hello".to_string()]],
            vec!["O".to_string()],
        );

        assert_eq!(instance.features.len(), 1);
        assert_eq!(instance.labels.len(), 1);
    }
}
