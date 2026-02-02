extern crate pyo3;
extern crate regex;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashSet;

pub mod crf;
pub mod featurizers;
pub mod lr;
pub mod text;

// Re-export CRF types
use crf::model::CRFModel;
use crf::serialization::{CRFFormat, ModelLoader, ModelSaver};
use crf::tagger::CRFTagger;
use crf::trainer::{CRFTrainer as RustCRFTrainer, LossFunction, TrainerConfig, TrainingInstance};

// Re-export LR types
use lr::model::LRModel;
use lr::predictor::LRClassifier;
use lr::serialization::{LRFormat, LRModelLoader, LRModelSaver};
use lr::trainer::{
    LRTrainer as RustLRTrainer, TrainerConfig as LRTrainerConfig,
    TrainingInstance as LRTrainingInstance,
};

// Re-export Text types
use text::tfidf::{TfIdfConfig, TfIdfVectorizer};

#[pyclass]
pub struct CRFFeaturizer {
    pub object: featurizers::CRFFeaturizer,
}

#[pymethods]
impl CRFFeaturizer {
    #[new]
    pub fn new(feature_configs: Vec<String>, dictionary: HashSet<String>) -> PyResult<Self> {
        Ok(CRFFeaturizer {
            object: featurizers::CRFFeaturizer::new(feature_configs, dictionary),
        })
    }

    pub fn process(
        self_: PyRef<Self>,
        sentences: Vec<Vec<Vec<String>>>,
    ) -> PyResult<Vec<Vec<Vec<String>>>> {
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

    /// Save the model to a file in CRFsuite format (compatible with python-crfsuite)
    pub fn save(&self, path: String) -> PyResult<()> {
        let saver = ModelSaver::new();
        saver
            .save(&self.model, path, CRFFormat::CRFsuite)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load a model from a file
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let loader = ModelLoader::new();
        let model = loader
            .load(path, CRFFormat::Auto)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
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
        self.tagger
            .load(path)
            .map_err(pyo3::exceptions::PyIOError::new_err)
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
        let labels: Vec<String> = result
            .labels
            .iter()
            .map(|&id| {
                self.tagger
                    .model()
                    .id_to_label(id)
                    .unwrap_or("O")
                    .to_string()
            })
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
    ///     loss_function: "lbfgs" (recommended), "nll" for SGD, or "perceptron" for Structured Perceptron
    ///     l1_penalty: L1 regularization penalty (for lbfgs and nll)
    ///     l2_penalty: L2 regularization penalty (for lbfgs and nll)
    ///     learning_rate: Learning rate (only for perceptron)
    ///     max_iterations: Maximum number of training iterations
    ///     averaging: Whether to use averaged perceptron (only for perceptron)
    ///     verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
    #[new]
    #[pyo3(signature = (loss_function="lbfgs", l1_penalty=0.0, l2_penalty=0.01, learning_rate=0.1, max_iterations=100, averaging=true, verbose=1))]
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
            "lbfgs"
            | "LBFGS"
            | "l-bfgs"
            | "L-BFGS"
            | "nll"
            | "NLL"
            | "negative_log_likelihood"
            | "sgd"
            | "SGD" => LossFunction::LBFGS {
                l1_penalty,
                l2_penalty,
            },
            "perceptron" | "Perceptron" | "structured_perceptron" => {
                LossFunction::StructuredPerceptron { learning_rate }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown loss function: {}. Use 'lbfgs' (recommended) or 'perceptron'",
                    loss_function
                )));
            }
        };

        let config = TrainerConfig {
            loss_function: loss,
            max_iterations,
            epsilon: 1e-5,
            averaging,
            verbose: verbose as i32,
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "X and y must have the same length: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        // Convert to training instances
        let data: Vec<TrainingInstance> = x
            .into_iter()
            .zip(y)
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

// ============================================================================
// Python bindings for LR classes
// ============================================================================

/// Python wrapper for LR Model
#[pyclass(name = "LRModel")]
pub struct PyLRModel {
    model: LRModel,
}

#[pymethods]
impl PyLRModel {
    /// Create a new empty LR model
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            model: LRModel::new(),
        })
    }

    /// Create a model with predefined classes
    #[staticmethod]
    pub fn with_classes(classes: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            model: LRModel::with_classes(classes),
        })
    }

    /// Get the number of classes
    #[getter]
    pub fn num_classes(&self) -> usize {
        self.model.num_classes
    }

    /// Get the number of features
    #[getter]
    pub fn num_features(&self) -> usize {
        self.model.num_features
    }

    /// Get the number of non-zero weights
    pub fn num_weights(&self) -> usize {
        self.model.num_weights()
    }

    /// Get all class labels
    pub fn get_classes(&self) -> Vec<String> {
        self.model.get_classes()
    }

    /// Save the model to a file
    pub fn save(&self, path: String) -> PyResult<()> {
        let saver = LRModelSaver::new();
        saver
            .save(&self.model, path, LRFormat::Native)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load a model from a file
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let loader = LRModelLoader::new();
        let model = loader
            .load(path, LRFormat::Auto)
            .map_err(pyo3::exceptions::PyIOError::new_err)?;
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
            "LRModel(num_classes={}, num_features={}, num_weights={})",
            self.model.num_classes,
            self.model.num_features,
            self.model.num_weights()
        )
    }
}

/// Python wrapper for LR Classifier
#[pyclass(name = "LRClassifier")]
pub struct PyLRClassifier {
    classifier: LRClassifier,
}

#[pymethods]
impl PyLRClassifier {
    /// Create a new classifier with an empty model
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            classifier: LRClassifier::new(),
        })
    }

    /// Create a classifier from a model
    #[staticmethod]
    pub fn from_model(model: &PyLRModel) -> PyResult<Self> {
        Ok(Self {
            classifier: LRClassifier::from_model(model.model.clone()),
        })
    }

    /// Load a model from file
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let classifier = LRClassifier::load(path).map_err(pyo3::exceptions::PyIOError::new_err)?;
        Ok(Self { classifier })
    }

    /// Predict the most likely class for the given features
    ///
    /// Args:
    ///     features: List of feature strings like "word=hello"
    ///
    /// Returns:
    ///     The predicted class label
    pub fn predict(&self, features: Vec<String>) -> String {
        self.classifier.predict(&features)
    }

    /// Predict with probability for the most likely class
    ///
    /// Returns:
    ///     Tuple of (class_label, probability)
    pub fn predict_with_prob(&self, features: Vec<String>) -> (String, f64) {
        self.classifier.predict_with_prob(&features)
    }

    /// Get probability distribution over all classes
    ///
    /// Returns:
    ///     List of (class_label, probability) tuples, sorted by probability descending
    pub fn predict_proba(&self, features: Vec<String>) -> Vec<(String, f64)> {
        self.classifier.predict_proba(&features)
    }

    /// Get top-k most likely classes with probabilities
    pub fn predict_top_k(&self, features: Vec<String>, k: usize) -> Vec<(String, f64)> {
        self.classifier.predict_top_k(&features, k)
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.classifier.num_classes()
    }

    /// Get all class labels
    pub fn classes(&self) -> Vec<String> {
        self.classifier.classes()
    }

    fn __repr__(&self) -> String {
        format!(
            "LRClassifier(num_classes={})",
            self.classifier.num_classes()
        )
    }
}

/// Python wrapper for LR Trainer
#[pyclass(name = "LRTrainer")]
pub struct PyLRTrainer {
    trainer: RustLRTrainer,
}

#[pymethods]
impl PyLRTrainer {
    /// Create a new trainer with configuration
    ///
    /// Args:
    ///     l1_penalty: L1 regularization penalty (lasso)
    ///     l2_penalty: L2 regularization penalty (ridge)
    ///     learning_rate: Learning rate for SGD
    ///     max_epochs: Maximum number of training epochs
    ///     batch_size: Mini-batch size (1 = pure SGD)
    ///     tol: Convergence tolerance for early stopping
    ///     verbose: Verbosity level (0=quiet, 1=progress, 2=detailed)
    #[new]
    #[pyo3(signature = (l1_penalty=0.0, l2_penalty=0.01, learning_rate=0.1, max_epochs=100, batch_size=1, tol=1e-4, verbose=1))]
    pub fn new(
        l1_penalty: f64,
        l2_penalty: f64,
        learning_rate: f64,
        max_epochs: usize,
        batch_size: usize,
        tol: f64,
        verbose: u8,
    ) -> PyResult<Self> {
        let config = LRTrainerConfig {
            l1_penalty,
            l2_penalty,
            learning_rate,
            max_epochs,
            batch_size: batch_size.max(1),
            tol,
            verbose,
        };

        Ok(Self {
            trainer: RustLRTrainer::with_config(config),
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

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.trainer.set_learning_rate(lr);
    }

    /// Set maximum epochs
    pub fn set_max_epochs(&mut self, epochs: usize) {
        self.trainer.set_max_epochs(epochs);
    }

    /// Set batch size
    pub fn set_batch_size(&mut self, size: usize) {
        self.trainer.set_batch_size(size);
    }

    /// Train the model on the given data
    ///
    /// Args:
    ///     X: List of feature lists, one per instance.
    ///        Each inner list contains feature strings like "word=hello".
    ///     y: List of class labels, one per instance.
    ///
    /// Returns:
    ///     Trained LRModel
    pub fn train(&mut self, x: Vec<Vec<String>>, y: Vec<String>) -> PyResult<PyLRModel> {
        if x.len() != y.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "X and y must have the same length: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        // Convert to training instances
        let data: Vec<LRTrainingInstance> = x
            .into_iter()
            .zip(y)
            .map(|(features, label)| LRTrainingInstance::new(features, label))
            .collect();

        // Train
        let model = self.trainer.train(&data);

        Ok(PyLRModel { model })
    }

    /// Get the current model
    pub fn get_model(&self) -> PyLRModel {
        PyLRModel {
            model: self.trainer.get_model().clone(),
        }
    }

    fn __repr__(&self) -> String {
        "LRTrainer()".to_string()
    }
}

// ============================================================================
// Python bindings for Text classes
// ============================================================================

/// Python wrapper for TF-IDF Vectorizer
#[pyclass(name = "TfIdfVectorizer")]
pub struct PyTfIdfVectorizer {
    vectorizer: TfIdfVectorizer,
}

#[pymethods]
impl PyTfIdfVectorizer {
    /// Create a new TfIdfVectorizer with optional configuration
    ///
    /// Args:
    ///     min_df: Minimum document frequency (default: 1)
    ///     max_df: Maximum document frequency ratio (default: 1.0)
    ///     max_features: Maximum vocabulary size, 0 for unlimited (default: 0)
    ///     sublinear_tf: Use sublinear TF scaling (default: False)
    ///     lowercase: Convert text to lowercase (default: True)
    ///     ngram_range: Tuple of (min_n, max_n) for n-grams (default: (1, 1))
    ///     min_token_length: Minimum token length (default: 2, matches sklearn)
    ///     norm: Apply L2 normalization (default: True)
    #[new]
    #[pyo3(signature = (min_df=1, max_df=1.0, max_features=0, sublinear_tf=false, lowercase=true, ngram_range=(1, 1), min_token_length=2, norm=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_df: usize,
        max_df: f64,
        max_features: usize,
        sublinear_tf: bool,
        lowercase: bool,
        ngram_range: (usize, usize),
        min_token_length: usize,
        norm: bool,
    ) -> PyResult<Self> {
        let config = TfIdfConfig {
            min_df,
            max_df,
            max_features,
            sublinear_tf,
            lowercase,
            ngram_range,
            min_token_length,
            norm,
        };
        Ok(Self {
            vectorizer: TfIdfVectorizer::with_config(config),
        })
    }

    /// Fit the vectorizer on a list of documents
    ///
    /// Args:
    ///     documents: List of text documents
    pub fn fit(&mut self, documents: Vec<String>) {
        self.vectorizer.fit(&documents);
    }

    /// Transform a document into sparse TF-IDF features
    ///
    /// Args:
    ///     document: Text document to transform
    ///
    /// Returns:
    ///     List of (feature_index, tfidf_value) tuples
    pub fn transform(&self, document: &str) -> Vec<(u32, f64)> {
        self.vectorizer.transform(document)
    }

    /// Transform a document into a dense TF-IDF vector
    ///
    /// Args:
    ///     document: Text document to transform
    ///
    /// Returns:
    ///     List of TF-IDF values (length = vocab_size)
    pub fn transform_dense(&self, document: &str) -> Vec<f64> {
        self.vectorizer.transform_dense(document)
    }

    /// Transform a document into feature strings for LRClassifier
    ///
    /// Args:
    ///     document: Text document to transform
    ///
    /// Returns:
    ///     List of feature strings like "tfidf_0=0.1234"
    pub fn transform_to_features(&self, document: &str) -> Vec<String> {
        self.vectorizer.transform_to_features(document)
    }

    /// Fit and transform documents in one step
    ///
    /// Args:
    ///     documents: List of text documents
    ///
    /// Returns:
    ///     List of sparse TF-IDF vectors
    pub fn fit_transform(&mut self, documents: Vec<String>) -> Vec<Vec<(u32, f64)>> {
        self.vectorizer.fit_transform(&documents)
    }

    /// Get the vocabulary size
    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.vectorizer.vocab_size()
    }

    /// Get the number of documents used for fitting
    #[getter]
    pub fn n_docs(&self) -> usize {
        self.vectorizer.n_docs()
    }

    /// Check if the vectorizer has been fitted
    pub fn is_fitted(&self) -> bool {
        self.vectorizer.is_fitted()
    }

    /// Get all feature names (vocabulary words in order)
    pub fn get_feature_names(&self) -> Vec<String> {
        self.vectorizer.get_feature_names()
    }

    /// Get the IDF values for all features
    pub fn get_idf(&self) -> Vec<f64> {
        self.vectorizer.idf_values().to_vec()
    }

    /// Get top features by IDF value
    ///
    /// Args:
    ///     n: Number of top features to return
    ///
    /// Returns:
    ///     List of (word, idf_value) tuples
    pub fn top_features_by_idf(&self, n: usize) -> Vec<(String, f64)> {
        self.vectorizer.top_features_by_idf(n)
    }

    /// Get the index of a word in the vocabulary
    ///
    /// Returns None if the word is not in the vocabulary
    pub fn get_index(&self, word: &str) -> Option<u32> {
        self.vectorizer.get_index(word)
    }

    /// Get the word at a given index
    ///
    /// Returns None if the index is out of bounds
    pub fn get_word(&self, index: u32) -> Option<String> {
        self.vectorizer.get_word(index).map(|s| s.to_string())
    }

    /// Save the vectorizer to a file
    pub fn save(&self, path: String) -> PyResult<()> {
        let data = bincode::serialize(&self.vectorizer).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Serialization error: {}", e))
        })?;
        std::fs::write(&path, data)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Write error: {}", e)))?;
        Ok(())
    }

    /// Load a vectorizer from a file
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        let data = std::fs::read(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e)))?;
        let vectorizer: TfIdfVectorizer = bincode::deserialize(&data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Deserialization error: {}", e))
        })?;
        Ok(Self { vectorizer })
    }

    fn __repr__(&self) -> String {
        if self.vectorizer.is_fitted() {
            format!(
                "TfIdfVectorizer(vocab_size={}, n_docs={})",
                self.vectorizer.vocab_size(),
                self.vectorizer.n_docs()
            )
        } else {
            "TfIdfVectorizer(not fitted)".to_string()
        }
    }
}

#[pymodule]
fn underthesea_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CRFFeaturizer>()?;
    m.add_class::<PyCRFModel>()?;
    m.add_class::<PyCRFTagger>()?;
    m.add_class::<PyCRFTrainer>()?;
    // LR classes
    m.add_class::<PyLRModel>()?;
    m.add_class::<PyLRClassifier>()?;
    m.add_class::<PyLRTrainer>()?;
    // Text classes
    m.add_class::<PyTfIdfVectorizer>()?;
    Ok(())
}
