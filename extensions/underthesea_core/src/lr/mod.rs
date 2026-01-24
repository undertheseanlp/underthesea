//! Logistic Regression (LR) module for multi-class text classification.
//!
//! This module provides a complete Logistic Regression implementation for training and inference,
//! supporting:
//! - Multi-class classification with softmax
//! - SGD training with mini-batches
//! - L1 and L2 regularization
//! - Sparse feature representation
//! - Native binary format serialization

pub mod model;
pub mod predictor;
pub mod serialization;
pub mod trainer;

// Re-export main types
pub use model::{ClassIndex, FeatureIndex, LRModel};
pub use predictor::LRClassifier;
pub use serialization::{LRFormat, LRModelLoader, LRModelSaver};
pub use trainer::{LRTrainer, TrainerConfig, TrainingInstance};
