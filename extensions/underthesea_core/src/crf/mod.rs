//! CRF (Conditional Random Field) module for sequence labeling.
//!
//! This module provides a complete CRF implementation for training and inference,
//! supporting:
//! - Viterbi decoding for inference
//! - Forward-backward algorithm for NLL training
//! - Structured perceptron training
//! - CRFsuite binary format compatibility

pub mod crfsuite_format;
pub mod features;
pub mod model;
pub mod serialization;
pub mod tagger;
pub mod trainer;

// Re-export main types
pub use features::{FeatureFunction, FeatureType};
pub use model::CRFModel;
pub use serialization::{CRFFormat, ModelLoader, ModelSaver};
pub use tagger::CRFTagger;
pub use trainer::{CRFTrainer, LossFunction, TrainerConfig};
