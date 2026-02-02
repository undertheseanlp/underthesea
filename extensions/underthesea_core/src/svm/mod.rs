//! Linear SVM - LIBLINEAR-style Dual Coordinate Descent
//!
//! Single unified class for training and inference.

use hashbrown::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Sparse feature vector
pub type SparseVec = Vec<(u32, f32)>;

/// Linear SVM Classifier (train + predict)
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct LinearSVC {
    weights_flat: Vec<f32>,
    biases: Vec<f32>,
    classes: Vec<String>,
    n_features: usize,
    n_classes: usize,
}

#[pymethods]
impl LinearSVC {
    #[new]
    pub fn new() -> Self {
        Self {
            weights_flat: Vec::new(),
            biases: Vec::new(),
            classes: Vec::new(),
            n_features: 0,
            n_classes: 0,
        }
    }

    /// Train the SVM classifier
    #[pyo3(signature = (features, labels, c=1.0, max_iter=1000, tol=0.1))]
    pub fn fit(
        &mut self,
        features: Vec<Vec<f64>>,
        labels: Vec<String>,
        c: f64,
        max_iter: usize,
        tol: f64,
    ) {
        let n_samples = features.len();
        let n_features = if n_samples > 0 { features[0].len() } else { 0 };

        // Convert to sparse format
        let sparse: Vec<SparseVec> = features
            .par_iter()
            .map(|dense| {
                dense
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v.abs() > 1e-10)
                    .map(|(i, &v)| (i as u32, v as f32))
                    .collect()
            })
            .collect();

        // Precompute ||x||^2
        let x_sq: Vec<f32> = sparse
            .par_iter()
            .map(|x| x.iter().map(|&(_, v)| v * v).sum())
            .collect();

        // Get unique classes
        let mut classes: Vec<String> = labels.iter().cloned().collect();
        classes.sort();
        classes.dedup();
        let n_classes = classes.len();

        let class_map: HashMap<String, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();

        let y_idx: Vec<usize> = labels.iter().map(|l| class_map[l]).collect();

        // Train OvR classifiers in parallel
        let results: Vec<(Vec<f32>, f32)> = (0..n_classes)
            .into_par_iter()
            .map(|cls| {
                let y: Vec<i8> = y_idx
                    .iter()
                    .map(|&i| if i == cls { 1 } else { -1 })
                    .collect();
                solve_svm(&sparse, &y, &x_sq, n_features, c as f32, tol as f32, max_iter)
            })
            .collect();

        // Store model
        self.n_features = n_features;
        self.n_classes = n_classes;
        self.classes = classes;
        self.biases = results.iter().map(|(_, b)| *b).collect();
        self.weights_flat = Vec::with_capacity(n_classes * n_features);
        for (w, _) in &results {
            self.weights_flat.extend_from_slice(w);
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> String {
        self.classes[self.argmax_f64(&features)].clone()
    }

    pub fn predict_batch(&self, batch: Vec<Vec<f64>>) -> Vec<String> {
        batch
            .par_iter()
            .map(|f| self.classes[self.argmax_f64(f)].clone())
            .collect()
    }

    #[getter]
    pub fn classes(&self) -> Vec<String> {
        self.classes.clone()
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        bincode::serialize_into(BufWriter::new(file), self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        bincode::deserialize_from(BufReader::new(file))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }
}

impl LinearSVC {
    #[inline(always)]
    fn weights(&self, cls: usize) -> &[f32] {
        let s = cls * self.n_features;
        &self.weights_flat[s..s + self.n_features]
    }

    #[inline]
    fn argmax_f64(&self, f: &[f64]) -> usize {
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let w = self.weights(c);
            let score: f32 = w.iter().zip(f).map(|(&wi, &fi)| wi * fi as f32).sum::<f32>()
                + self.biases[c];
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }

    /// Internal sparse prediction (for TextClassifier)
    #[inline]
    pub fn predict_sparse_idx(&self, f: &[(u32, f32)]) -> usize {
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let w = self.weights(c);
            let mut score = self.biases[c];
            for &(j, v) in f {
                score += unsafe { *w.get_unchecked(j as usize) } * v;
            }
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        best
    }

    #[inline]
    pub fn predict_sparse_internal(&self, f: &[(u32, f32)]) -> String {
        self.classes[self.predict_sparse_idx(f)].clone()
    }

    #[inline]
    pub fn predict_sparse_with_score_internal(&self, f: &[(u32, f32)]) -> (String, f64) {
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let w = self.weights(c);
            let mut score = self.biases[c];
            for &(j, v) in f {
                score += w[j as usize] * v;
            }
            if score > best_score {
                best_score = score;
                best = c;
            }
        }
        let conf = 1.0 / (1.0 + (-best_score as f64).exp());
        (self.classes[best].clone(), conf)
    }

    pub fn from_weights(
        weights: Vec<Vec<f32>>,
        biases: Vec<f32>,
        classes: Vec<String>,
        n_features: usize,
    ) -> Self {
        let n_classes = classes.len();
        let mut weights_flat = Vec::with_capacity(n_classes * n_features);
        for w in &weights {
            weights_flat.extend_from_slice(w);
        }
        Self { weights_flat, biases, classes, n_features, n_classes }
    }
}

/// LIBLINEAR L2-regularized L2-loss SVC (Dual Coordinate Descent)
#[inline(never)]
fn solve_svm(
    x: &[SparseVec],
    y: &[i8],
    x_sq: &[f32],
    n_features: usize,
    c: f32,
    eps: f32,
    max_iter: usize,
) -> (Vec<f32>, f32) {
    let n = x.len();
    let diag = 0.5 / c;
    let qd: Vec<f32> = x_sq.iter().map(|&xn| xn + diag).collect();

    let mut alpha = vec![0.0f32; n];
    let mut w = vec![0.0f32; n_features];
    let mut idx: Vec<usize> = (0..n).collect();

    for iter in 0..max_iter {
        // Shuffle
        for i in 0..n {
            let j = i + (iter * 1103515245 + 12345) % (n - i).max(1);
            idx.swap(i, j);
        }

        let mut max_viol = 0.0f32;

        for &i in &idx {
            let yi = y[i] as f32;
            let xi = &x[i];

            let wxi: f32 = xi.iter().map(|&(j, v)| w[j as usize] * v).sum();
            let g = yi * wxi - 1.0 + diag * alpha[i];
            let pg = if alpha[i] == 0.0 { g.min(0.0) } else { g };

            max_viol = max_viol.max(pg.abs());

            if pg.abs() > 1e-12 {
                let alpha_old = alpha[i];
                alpha[i] = (alpha[i] - g / qd[i]).max(0.0);
                let d = (alpha[i] - alpha_old) * yi;
                if d.abs() > 1e-12 {
                    for &(j, v) in xi {
                        w[j as usize] += d * v;
                    }
                }
            }
        }

        if max_viol <= eps {
            break;
        }
    }

    // Compute bias
    let mut bias_sum = 0.0f32;
    let mut n_sv = 0;
    for i in 0..n {
        if alpha[i] > 1e-8 {
            let yi = y[i] as f32;
            let wxi: f32 = x[i].iter().map(|&(j, v)| w[j as usize] * v).sum();
            bias_sum += yi * (1.0 - alpha[i] * diag) - wxi;
            n_sv += 1;
        }
    }
    let bias = if n_sv > 0 { bias_sum / n_sv as f32 } else { 0.0 };

    (w, bias)
}
