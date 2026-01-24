extern crate regex;
extern crate pyo3;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashSet;

pub mod featurizers;
pub mod crf;

// Re-export CRF types
use crf::model::CRFModel;
use crf::serialization::{CRFFormat, ModelLoader, ModelSaver};
use crf::tagger::CRFTagger;
use crf::trainer::{CRFTrainer as RustCRFTrainer, LossFunction, TrainerConfig, TrainingInstance};

#[pyclass]
pub struct CRFFeaturizer {
    pub object: featurizers::CRFFeaturizer
}


#[pymethods]
impl CRFFeaturizer {
    #[new]
    pub fn new(feature_configs: Vec<String>, dictionary: HashSet<String>) -> PyResult<Self> {
        Ok(CRFFeaturizer {
            object: featurizers::CRFFeaturizer::new(feature_configs, dictionary)
        })
    }

    pub fn process(self_: PyRef<Self>, sentences: Vec<Vec<Vec<String>>>) -> PyResult<Vec<Vec<Vec<String>>>> {
        let output = self_.object.process(sentences);
        Ok(output)
    }
}

// ============================================================================
// Python bindings for CRF classes
// ============================================================================

/// Python wrapper for CRF Model
#[pyclass(name = "CRFModel")]
pub struct PyCRFModel {
    model: CRFModel,
}

#[pymethods]
impl PyCRFModel {
    /// Create a new empty CRF model
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            model: CRFModel::new(),
        })
    }

    /// Create a model with predefined labels
    #[staticmethod]
    pub fn with_labels(labels: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            model: CRFModel::with_labels(labels),
        })
    }

    /// Get the number of labels
    #[getter]
    pub fn num_labels(&self) -> usize {
        self.model.num_labels
    }

    /// Get the number of attributes
    #[getter]
    pub fn num_attributes(&self) -> usize {
        self.model.num_attributes
    }

    /// Get the number of state features
    pub fn num_state_features(&self) -> usize {
        self.model.num_state_features()
    }

    /// Get the number of transition features
    pub fn num_transition_features(&self) -> usize {
        self.model.num_transition_features()
    }

    /// Get all label names
    pub fn get_labels(&self) -> Vec<String> {
        self.model.labels.labels().to_vec()
    }

    /// Save the model to a file
    pub fn save(&self, path: String) -> PyResult<()> {
        let saver = ModelSaver::new();
        saver.save(&self.model, path, CRFFormat::Native)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Load a model from a file
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let loader = ModelLoader::new();
        let model = loader.load(path, CRFFormat::Auto)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        Ok(Self { model })
    }

    /// Get the L2 norm squared of all weights
    pub fn l2_norm_squared(&self) -> f64 {
        self.model.l2_norm_squared()
    }

    /// Get the L1 norm of all weights
    pub fn l1_norm(&self) -> f64 {
        self.model.l1_norm()
    }

    fn __repr__(&self) -> String {
        format!(
            "CRFModel(num_labels={}, num_attributes={}, state_features={}, transition_features={})",
            self.model.num_labels,
            self.model.num_attributes,
            self.model.num_state_features(),
            self.model.num_transition_features()
        )
    }
}

/// Python wrapper for CRF Tagger
#[pyclass(name = "CRFTagger")]
pub struct PyCRFTagger {
    tagger: CRFTagger,
}

#[pymethods]
impl PyCRFTagger {
    /// Create a new tagger with an empty model
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            tagger: CRFTagger::new(),
        })
    }

    /// Create a tagger from a model
    #[staticmethod]
    pub fn from_model(model: &PyCRFModel) -> PyResult<Self> {
        Ok(Self {
            tagger: CRFTagger::from_model(model.model.clone()),
        })
    }

    /// Load a model from file
    pub fn load(&mut self, path: String) -> PyResult<()> {
        self.tagger.load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Tag a sequence of observations
    ///
    /// Args:
    ///     features: List of feature lists, one per token.
    ///               Each inner list contains feature strings like "word=hello".
    ///
    /// Returns:
    ///     List of label strings
    pub fn tag(&self, features: Vec<Vec<String>>) -> Vec<String> {
        self.tagger.tag(&features)
    }

    /// Tag a sequence and return the score along with labels
    pub fn tag_with_score(&self, features: Vec<Vec<String>>) -> (Vec<String>, f64) {
        let result = self.tagger.tag_with_score(&features);
        let labels: Vec<String> = result.labels
            .iter()
            .map(|&id| self.tagger.model().id_to_label(id).unwrap_or("O").to_string())
            .collect();
        (labels, result.score)
    }

    /// Compute marginal probabilities for each position and label
    pub fn marginals(&self, features: Vec<Vec<String>>) -> Vec<Vec<f64>> {
        self.tagger.compute_marginals(&features)
    }

    /// Get the number of labels
    pub fn num_labels(&self) -> usize {
        self.tagger.num_labels()
    }

    /// Get all label names
    pub fn labels(&self) -> Vec<String> {
        self.tagger.labels()
    }

    fn __repr__(&self) -> String {
        format!("CRFTagger(num_labels={})", self.tagger.num_labels())
    }
}

/// Python wrapper for CRF Trainer
#[pyclass(name = "CRFTrainer")]
pub struct PyCRFTrainer {
    trainer: RustCRFTrainer,
}

#[pymethods]
impl PyCRFTrainer {
    /// Create a new trainer with default configuration
    ///
    /// Args:
    ///     loss_function: "nll" for Negative Log-Likelihood or "perceptron" for Structured Perceptron
    ///     l1_penalty: L1 regularization penalty (only for NLL)
    ///     l2_penalty: L2 regularization penalty (only for NLL)
    ///     learning_rate: Learning rate (only for perceptron)
    ///     max_iterations: Maximum number of training iterations
    ///     averaging: Whether to use averaged perceptron (only for perceptron)
    ///     verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
    #[new]
    #[pyo3(signature = (loss_function="nll", l1_penalty=0.0, l2_penalty=0.01, learning_rate=0.1, max_iterations=100, averaging=true, verbose=1))]
    pub fn new(
        loss_function: &str,
        l1_penalty: f64,
        l2_penalty: f64,
        learning_rate: f64,
        max_iterations: usize,
        averaging: bool,
        verbose: u8,
    ) -> PyResult<Self> {
        let loss = match loss_function {
            "nll" | "NLL" | "negative_log_likelihood" => {
                LossFunction::NegativeLogLikelihood { l1_penalty, l2_penalty }
            }
            "perceptron" | "Perceptron" | "structured_perceptron" => {
                LossFunction::StructuredPerceptron { learning_rate }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown loss function: {}. Use 'nll' or 'perceptron'", loss_function)
                ));
            }
        };

        let config = TrainerConfig {
            loss_function: loss,
            max_iterations,
            epsilon: 1e-5,
            averaging,
            verbose,
        };

        Ok(Self {
            trainer: RustCRFTrainer::with_config(config),
        })
    }

    /// Set L1 regularization penalty
    pub fn set_l1_penalty(&mut self, penalty: f64) {
        self.trainer.set_l1_penalty(penalty);
    }

    /// Set L2 regularization penalty
    pub fn set_l2_penalty(&mut self, penalty: f64) {
        self.trainer.set_l2_penalty(penalty);
    }

    /// Set maximum iterations
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.trainer.set_max_iterations(max_iter);
    }

    /// Train the model on the given data
    ///
    /// Args:
    ///     X: List of sequences, where each sequence is a list of feature lists
    ///        (one feature list per token)
    ///     y: List of label sequences, where each sequence is a list of label strings
    ///
    /// Returns:
    ///     Trained CRFModel
    pub fn train(&mut self, x: Vec<Vec<Vec<String>>>, y: Vec<Vec<String>>) -> PyResult<PyCRFModel> {
        if x.len() != y.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("X and y must have the same length: {} vs {}", x.len(), y.len())
            ));
        }

        // Convert to training instances
        let data: Vec<TrainingInstance> = x.into_iter()
            .zip(y.into_iter())
            .map(|(features, labels)| TrainingInstance::new(features, labels))
            .collect();

        // Train
        let model = self.trainer.train(&data);

        Ok(PyCRFModel { model })
    }

    /// Get the current model (during or after training)
    pub fn get_model(&self) -> PyCRFModel {
        PyCRFModel {
            model: self.trainer.get_model().clone(),
        }
    }

    fn __repr__(&self) -> String {
        "CRFTrainer()".to_string()
    }
}

#[pymodule]
fn underthesea_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CRFFeaturizer>()?;
    m.add_class::<PyCRFModel>()?;
    m.add_class::<PyCRFTagger>()?;
    m.add_class::<PyCRFTrainer>()?;
    Ok(())
}
