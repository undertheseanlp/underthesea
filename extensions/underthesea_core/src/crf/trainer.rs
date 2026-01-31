//! CRF Training algorithms.
//!
//! This module provides training algorithms for CRF models:
//! - L-BFGS with OWL-QN (Orthant-Wise Limited-memory Quasi-Newton) for L1 regularization
//! - L-BFGS for L2-only regularization
//! - Structured Perceptron (online learning)
//!
//! L-BFGS/OWL-QN is recommended for best performance, matching CRFsuite speed.

#![allow(clippy::needless_range_loop)]

use super::model::CRFModel;
use super::tagger::CRFTagger;
use hashbrown::HashMap;
use liblbfgs::lbfgs;
use serde::{Deserialize, Serialize};

/// Loss function types for CRF training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    /// L-BFGS optimization (fast, recommended)
    LBFGS { l1_penalty: f64, l2_penalty: f64 },

    /// Structured Perceptron (online learning)
    StructuredPerceptron { learning_rate: f64 },
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::LBFGS {
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

    /// Convergence threshold
    pub epsilon: f64,

    /// Verbose output level (0 = silent, 1 = progress, 2 = debug)
    pub verbose: i32,

    /// Use averaged weights for perceptron
    pub averaging: bool,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            loss_function: LossFunction::default(),
            max_iterations: 100,
            epsilon: 1e-5,
            verbose: 1,
            averaging: true,
        }
    }
}

/// Training instance: sequence of feature vectors and labels.
#[derive(Debug, Clone)]
pub struct TrainingInstance {
    pub features: Vec<Vec<String>>,
    pub labels: Vec<String>,
}

impl TrainingInstance {
    pub fn new(features: Vec<Vec<String>>, labels: Vec<String>) -> Self {
        Self { features, labels }
    }
}

/// CRF Trainer using L-BFGS optimization.
pub struct CRFTrainer {
    model: CRFModel,
    tagger: CRFTagger,
    config: TrainerConfig,
    update_count: usize,
    #[allow(dead_code)]
    avg_state_weights: HashMap<(u32, u32), f64>,
    #[allow(dead_code)]
    avg_transition_weights: Vec<f64>,
}

impl CRFTrainer {
    pub fn new() -> Self {
        let model = CRFModel::new();
        let tagger = CRFTagger::from_model(model.clone());
        Self {
            model,
            tagger,
            config: TrainerConfig::default(),
            update_count: 0,
            avg_state_weights: HashMap::new(),
            avg_transition_weights: Vec::new(),
        }
    }

    pub fn with_config(config: TrainerConfig) -> Self {
        let model = CRFModel::new();
        let tagger = CRFTagger::from_model(model.clone());
        Self {
            model,
            tagger,
            config,
            update_count: 0,
            avg_state_weights: HashMap::new(),
            avg_transition_weights: Vec::new(),
        }
    }

    pub fn set_verbose(&mut self, verbose: i32) {
        self.config.verbose = verbose;
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.config.max_iterations = max_iter;
    }

    pub fn set_l1_penalty(&mut self, penalty: f64) {
        if let LossFunction::LBFGS { l2_penalty, .. } = self.config.loss_function {
            self.config.loss_function = LossFunction::LBFGS {
                l1_penalty: penalty,
                l2_penalty,
            };
        }
    }

    pub fn set_l2_penalty(&mut self, penalty: f64) {
        if let LossFunction::LBFGS { l1_penalty, .. } = self.config.loss_function {
            self.config.loss_function = LossFunction::LBFGS {
                l1_penalty,
                l2_penalty: penalty,
            };
        }
    }

    pub fn train(&mut self, data: &[TrainingInstance]) -> CRFModel {
        self.initialize_from_data(data);

        match self.config.loss_function.clone() {
            LossFunction::LBFGS {
                l1_penalty,
                l2_penalty,
            } => {
                self.train_lbfgs(data, l1_penalty, l2_penalty);
            }
            LossFunction::StructuredPerceptron { learning_rate } => {
                self.train_perceptron(data, learning_rate);
            }
        }

        self.model.clone()
    }

    fn initialize_from_data(&mut self, data: &[TrainingInstance]) {
        for instance in data {
            for label in &instance.labels {
                self.model.label_to_id(label);
            }
        }

        for instance in data {
            for token_features in &instance.features {
                for attr in token_features {
                    self.model.attributes.get_or_insert(attr);
                }
            }
        }

        self.model.num_attributes = self.model.attributes.len();
        self.tagger = CRFTagger::from_model(self.model.clone());

        if self.config.averaging {
            self.avg_transition_weights = vec![0.0; self.model.num_labels * self.model.num_labels];
        }

        if self.config.verbose > 0 {
            eprintln!(
                "Initialized model with {} labels and {} attributes",
                self.model.num_labels, self.model.num_attributes
            );
        }
    }

    /// Train using L-BFGS optimization with OWL-QN for L1 regularization.
    fn train_lbfgs(&mut self, data: &[TrainingInstance], l1_penalty: f64, l2_penalty: f64) {
        let num_labels = self.model.num_labels;
        let num_attrs = self.model.num_attributes;

        // Precompute attribute IDs and label IDs for all instances
        let instances: Vec<(Vec<Vec<u32>>, Vec<u32>)> = data
            .iter()
            .filter_map(|inst| {
                let attr_ids: Vec<Vec<u32>> = inst
                    .features
                    .iter()
                    .map(|f| self.model.attrs_to_ids_readonly(f))
                    .collect();

                let label_ids: Vec<u32> = inst
                    .labels
                    .iter()
                    .filter_map(|l| self.model.label_to_id_readonly(l))
                    .collect();

                if label_ids.len() == inst.labels.len() {
                    Some((attr_ids, label_ids))
                } else {
                    None
                }
            })
            .collect();

        if instances.is_empty() {
            eprintln!("No valid training instances!");
            return;
        }

        // SPARSE FEATURE OPTIMIZATION
        let mut active_features: HashMap<(u32, u32), u32> = HashMap::new();
        let mut feature_to_attr_label: Vec<(u32, u32)> = Vec::new();
        let mut empirical_counts: Vec<f64> = Vec::new();

        let mut temp_counts: HashMap<(u32, u32), f64> = HashMap::new();
        for (attr_ids, gold_labels) in &instances {
            for (t, &label) in gold_labels.iter().enumerate() {
                for &attr_id in &attr_ids[t] {
                    *temp_counts.entry((attr_id, label)).or_insert(0.0) += 1.0;
                }
            }
        }

        for ((attr_id, label_id), count) in temp_counts {
            let feature_id = active_features.len() as u32;
            active_features.insert((attr_id, label_id), feature_id);
            feature_to_attr_label.push((attr_id, label_id));
            empirical_counts.push(count);
        }

        let num_state_features = active_features.len();
        let num_trans_params = num_labels * num_labels;
        let num_params = num_state_features + num_trans_params;

        let mut empirical_trans = vec![0.0f64; num_trans_params];
        for (_, gold_labels) in &instances {
            for t in 1..gold_labels.len() {
                let idx = gold_labels[t - 1] as usize * num_labels + gold_labels[t] as usize;
                empirical_trans[idx] += 1.0;
            }
        }

        if self.config.verbose > 0 {
            eprintln!(
                "Sparse features: {} state + {} trans = {} total (vs {} dense)",
                num_state_features,
                num_trans_params,
                num_params,
                num_attrs * num_labels + num_trans_params
            );
        }

        // Build FLAT attr -> features lookup
        let mut attr_to_features_temp: Vec<Vec<(u32, u32)>> = vec![Vec::new(); num_attrs];
        for (&(attr_id, label_id), &feature_id) in &active_features {
            attr_to_features_temp[attr_id as usize].push((label_id, feature_id));
        }

        let mut attr_offsets: Vec<u32> = Vec::with_capacity(num_attrs + 1);
        let mut attr_features_flat: Vec<(u32, u32)> = Vec::with_capacity(active_features.len());
        let mut offset = 0u32;
        for features in &attr_to_features_temp {
            attr_offsets.push(offset);
            for &f in features {
                attr_features_flat.push(f);
            }
            offset += features.len() as u32;
        }
        attr_offsets.push(offset);
        drop(attr_to_features_temp);

        let mut params = vec![0.0f64; num_params];

        let verbose = self.config.verbose;
        let max_iterations = self.config.max_iterations;

        let use_owlqn = l1_penalty > 0.0;
        let effective_l1 = if use_owlqn { 0.0 } else { l1_penalty };
        let eval_count = std::cell::RefCell::new(0usize);

        let evaluate = |x: &[f64], gx: &mut [f64]| {
            *eval_count.borrow_mut() += 1;
            let fx = compute_objective_and_gradient_sparse(
                x,
                gx,
                &instances,
                &empirical_counts,
                &empirical_trans,
                &attr_offsets,
                &attr_features_flat,
                num_labels,
                num_state_features,
                effective_l1,
                l2_penalty,
            );
            Ok(fx)
        };

        let mut iteration = 0usize;
        let progress = |prgr: &liblbfgs::Progress| -> bool {
            iteration = prgr.niter;
            if verbose > 0 {
                let num_active = prgr.x.iter().filter(|&&w| w != 0.0).count();
                eprintln!(
                    "***** Iteration #{} *****\nLoss: {:.6}\nFeature norm: {:.6}\nError norm: {:.6}\nActive features: {}\nLine search trials: {}\nLine search step: {:.6}",
                    iteration, prgr.fx, prgr.xnorm, prgr.gnorm, num_active, prgr.ncall, prgr.step
                );
            }
            iteration >= max_iterations
        };

        if verbose > 0 {
            if use_owlqn {
                eprintln!(
                    "Starting OWL-QN optimization with {} parameters (L1={}, L2={})...",
                    num_params, l1_penalty, l2_penalty
                );
            } else {
                eprintln!(
                    "Starting L-BFGS optimization with {} parameters (L2={})...",
                    num_params, l2_penalty
                );
            }
        }

        // Match CRFsuite configuration: MoreThuente line search
        let result = if use_owlqn {
            lbfgs()
                .with_max_iterations(max_iterations)
                .with_epsilon(1e-5)
                .with_fx_delta(1e-5, 10)
                .with_max_linesearch(20)
                .with_max_step_size(1e20)
                .with_linesearch_algorithm("Backtracking") // Required for OWL-QN
                .with_orthantwise(l1_penalty, 0, num_params)
                .minimize(&mut params, evaluate, progress)
        } else {
            lbfgs()
                .with_max_iterations(max_iterations)
                .with_epsilon(1e-5)
                .with_fx_delta(1e-5, 10)
                .with_max_linesearch(20)
                .with_max_step_size(1e20)
                .with_linesearch_algorithm("MoreThuente")
                .minimize(&mut params, evaluate, progress)
        };

        if verbose > 0 {
            match &result {
                Ok(report) => eprintln!(
                    "OWL-QN converged with final loss: {:.6} ({} evaluations)",
                    report.fx,
                    eval_count.borrow()
                ),
                Err(e) => eprintln!("L-BFGS failed: {:?}", e),
            }
        }

        // Copy weights back to model
        for (feature_id, &(attr_id, label_id)) in feature_to_attr_label.iter().enumerate() {
            let weight = params[feature_id];
            if weight != 0.0 {
                self.model.set_state_weight(attr_id, label_id, weight);
            }
        }

        for from_label in 0..num_labels {
            for to_label in 0..num_labels {
                let idx = num_state_features + from_label * num_labels + to_label;
                let weight = params[idx];
                if weight != 0.0 {
                    self.model
                        .set_transition(from_label as u32, to_label as u32, weight);
                }
            }
        }

        self.tagger = CRFTagger::from_model(self.model.clone());
    }

    fn train_perceptron(&mut self, data: &[TrainingInstance], learning_rate: f64) {
        let mut num_errors;
        let mut prev_errors = usize::MAX;

        for iter in 0..self.config.max_iterations {
            num_errors = 0;

            for instance in data {
                if self.perceptron_update(instance, learning_rate) {
                    num_errors += 1;
                }
                self.update_count += 1;
            }

            if self.config.verbose > 0 && (iter + 1) % 10 == 0 {
                eprintln!(
                    "Iteration {}: {} errors ({:.2}%)",
                    iter + 1,
                    num_errors,
                    100.0 * num_errors as f64 / data.len() as f64
                );
            }

            if num_errors == 0 {
                if self.config.verbose > 0 {
                    eprintln!("Converged at iteration {}", iter + 1);
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

            self.tagger = CRFTagger::from_model(self.model.clone());
        }

        if self.config.averaging && self.update_count > 0 {
            self.apply_average();
        }
    }

    fn perceptron_update(&mut self, instance: &TrainingInstance, learning_rate: f64) -> bool {
        let n = instance.features.len();
        if n == 0 {
            return false;
        }

        let predicted = self.tagger.tag(&instance.features);

        if predicted == instance.labels {
            return false;
        }

        for (t, (gold, pred)) in instance.labels.iter().zip(predicted.iter()).enumerate() {
            if gold != pred {
                let gold_id = self.model.label_to_id(gold);
                let pred_id = self.model.label_to_id(pred);
                for attr in &instance.features[t] {
                    let attr_id = self.model.attributes.get_or_insert(attr);
                    self.model.add_state_weight(attr_id, gold_id, learning_rate);
                    self.model
                        .add_state_weight(attr_id, pred_id, -learning_rate);
                }
            }
        }

        for t in 1..n {
            let gold_prev = self.model.label_to_id(&instance.labels[t - 1]);
            let gold_curr = self.model.label_to_id(&instance.labels[t]);
            let pred_prev = self.model.label_to_id(&predicted[t - 1]);
            let pred_curr = self.model.label_to_id(&predicted[t]);

            if gold_prev != pred_prev || gold_curr != pred_curr {
                self.model
                    .add_transition(gold_prev, gold_curr, learning_rate);
                self.model
                    .add_transition(pred_prev, pred_curr, -learning_rate);
            }
        }

        true
    }

    fn apply_average(&mut self) {
        // Simplified - just use current weights
    }

    pub fn get_model(&self) -> &CRFModel {
        &self.model
    }
}

impl Default for CRFTrainer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// L-BFGS Objective and Gradient Computation
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn compute_objective_and_gradient_sparse(
    params: &[f64],
    gradient: &mut [f64],
    instances: &[(Vec<Vec<u32>>, Vec<u32>)],
    empirical_state: &[f64],
    empirical_trans: &[f64],
    attr_offsets: &[u32],
    attr_features_flat: &[(u32, u32)],
    num_labels: usize,
    num_state_features: usize,
    l1_penalty: f64,
    l2_penalty: f64,
) -> f64 {
    // Initialize gradient = -empirical_count
    for (i, &emp) in empirical_state.iter().enumerate() {
        gradient[i] = -emp;
    }
    for (i, &emp) in empirical_trans.iter().enumerate() {
        gradient[num_state_features + i] = -emp;
    }

    // Precompute exp(transition_scores)
    let mut exp_trans = vec![0.0f64; num_labels * num_labels];
    for i in 0..num_labels {
        for j in 0..num_labels {
            let idx = num_state_features + i * num_labels + j;
            exp_trans[i * num_labels + j] = params[idx].exp();
        }
    }

    // Sequential processing - optimized with pre-allocated buffers
    let max_len = instances.iter().map(|(a, _)| a.len()).max().unwrap_or(0);

    let mut alpha = vec![0.0f64; max_len * num_labels];
    let mut beta = vec![0.0f64; max_len * num_labels];
    let mut exp_state = vec![0.0f64; max_len * num_labels];
    let mut scale_factors = vec![0.0f64; max_len];
    let mut state_mexp = vec![0.0f64; max_len * num_labels];
    let mut trans_mexp = vec![0.0f64; num_labels * num_labels];

    let mut total_loss = 0.0f64;
    for (attr_ids, gold_labels) in instances {
        let n = attr_ids.len();
        if n == 0 {
            continue;
        }
        total_loss += compute_instance_scaled(
            params,
            gradient,
            attr_ids,
            gold_labels,
            attr_offsets,
            attr_features_flat,
            &exp_trans,
            num_labels,
            num_state_features,
            &mut alpha[..n * num_labels],
            &mut beta[..n * num_labels],
            &mut exp_state[..n * num_labels],
            &mut scale_factors[..n],
            &mut state_mexp[..n * num_labels],
            &mut trans_mexp,
        );
    }

    // L2 regularization
    if l2_penalty > 0.0 {
        let two_l2 = 2.0 * l2_penalty;
        for i in 0..params.len() {
            let w = params[i];
            total_loss += l2_penalty * w * w;
            gradient[i] += two_l2 * w;
        }
    }

    // L1 regularization
    if l1_penalty > 0.0 {
        for (i, &w) in params.iter().enumerate() {
            total_loss += l1_penalty * w.abs();
            if w != 0.0 {
                gradient[i] += l1_penalty * w.signum();
            }
        }
    }

    total_loss
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_instance_scaled(
    params: &[f64],
    gradient: &mut [f64],
    attr_ids: &[Vec<u32>],
    gold_labels: &[u32],
    attr_offsets: &[u32],
    attr_features_flat: &[(u32, u32)],
    exp_trans: &[f64],
    num_labels: usize,
    num_state_features: usize,
    alpha: &mut [f64],
    beta: &mut [f64],
    exp_state: &mut [f64],
    scale_factors: &mut [f64],
    state_mexp: &mut [f64],
    trans_mexp: &mut [f64],
) -> f64 {
    let n = attr_ids.len();
    let num_attrs = attr_offsets.len().saturating_sub(1);
    let chunks = num_labels / 4;

    // Reset buffers
    exp_state.iter_mut().for_each(|e| *e = 0.0);
    trans_mexp.iter_mut().for_each(|e| *e = 0.0);

    // Compute exp(state_scores) with flat lookup
    unsafe {
        for t in 0..n {
            let base = t * num_labels;
            for &attr_id in &attr_ids[t] {
                let attr_idx = attr_id as usize;
                if attr_idx < num_attrs {
                    let start = *attr_offsets.get_unchecked(attr_idx) as usize;
                    let end = *attr_offsets.get_unchecked(attr_idx + 1) as usize;
                    for i in start..end {
                        let (label_id, feature_id) = *attr_features_flat.get_unchecked(i);
                        *exp_state.get_unchecked_mut(base + label_id as usize) +=
                            *params.get_unchecked(feature_id as usize);
                    }
                }
            }
            for y in 0..num_labels {
                let idx = base + y;
                *exp_state.get_unchecked_mut(idx) = exp_state.get_unchecked(idx).exp();
            }
        }
    }

    // Forward pass
    alpha[..num_labels].copy_from_slice(&exp_state[..num_labels]);
    let sum: f64 = alpha[..num_labels].iter().sum();
    let scale0 = if sum != 0.0 { 1.0 / sum } else { 1.0 };
    scale_factors[0] = scale0;
    for y in 0..num_labels {
        alpha[y] *= scale0;
    }

    // Forward recursion with loop unrolling
    unsafe {
        for t in 1..n {
            let curr_base = t * num_labels;
            let prev_base = (t - 1) * num_labels;

            for y in 0..num_labels {
                *alpha.get_unchecked_mut(curr_base + y) = 0.0;
            }

            for y_prev in 0..num_labels {
                let alpha_prev = *alpha.get_unchecked(prev_base + y_prev);
                let trans_base = y_prev * num_labels;

                for i in 0..chunks {
                    let y = i * 4;
                    let a0 = *alpha.get_unchecked(curr_base + y);
                    let a1 = *alpha.get_unchecked(curr_base + y + 1);
                    let a2 = *alpha.get_unchecked(curr_base + y + 2);
                    let a3 = *alpha.get_unchecked(curr_base + y + 3);

                    let t0 = *exp_trans.get_unchecked(trans_base + y);
                    let t1 = *exp_trans.get_unchecked(trans_base + y + 1);
                    let t2 = *exp_trans.get_unchecked(trans_base + y + 2);
                    let t3 = *exp_trans.get_unchecked(trans_base + y + 3);

                    *alpha.get_unchecked_mut(curr_base + y) = a0 + alpha_prev * t0;
                    *alpha.get_unchecked_mut(curr_base + y + 1) = a1 + alpha_prev * t1;
                    *alpha.get_unchecked_mut(curr_base + y + 2) = a2 + alpha_prev * t2;
                    *alpha.get_unchecked_mut(curr_base + y + 3) = a3 + alpha_prev * t3;
                }
                for y in (chunks * 4)..num_labels {
                    *alpha.get_unchecked_mut(curr_base + y) +=
                        alpha_prev * *exp_trans.get_unchecked(trans_base + y);
                }
            }

            let mut sum = 0.0f64;
            for y in 0..num_labels {
                let val =
                    *alpha.get_unchecked(curr_base + y) * *exp_state.get_unchecked(curr_base + y);
                *alpha.get_unchecked_mut(curr_base + y) = val;
                sum += val;
            }

            let scale = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            *scale_factors.get_unchecked_mut(t) = scale;
            for y in 0..num_labels {
                *alpha.get_unchecked_mut(curr_base + y) *= scale;
            }
        }
    }

    let log_norm: f64 = -scale_factors[..n].iter().map(|s| s.ln()).sum::<f64>();

    // Backward pass
    let last_base = (n - 1) * num_labels;
    let last_scale = scale_factors[n - 1];
    for y in 0..num_labels {
        beta[last_base + y] = last_scale;
    }

    // Backward pass
    unsafe {
        for t in (0..n - 1).rev() {
            let curr_base = t * num_labels;
            let next_base = (t + 1) * num_labels;
            let scale = *scale_factors.get_unchecked(t);

            for y in 0..num_labels {
                *state_mexp.get_unchecked_mut(y) =
                    *exp_state.get_unchecked(next_base + y) * *beta.get_unchecked(next_base + y);
            }

            for y in 0..num_labels {
                let trans_base = y * num_labels;
                let mut sum = 0.0f64;

                for i in 0..chunks {
                    let j = i * 4;
                    sum += *exp_trans.get_unchecked(trans_base + j) * *state_mexp.get_unchecked(j);
                    sum += *exp_trans.get_unchecked(trans_base + j + 1)
                        * *state_mexp.get_unchecked(j + 1);
                    sum += *exp_trans.get_unchecked(trans_base + j + 2)
                        * *state_mexp.get_unchecked(j + 2);
                    sum += *exp_trans.get_unchecked(trans_base + j + 3)
                        * *state_mexp.get_unchecked(j + 3);
                }
                for j in (chunks * 4)..num_labels {
                    sum += *exp_trans.get_unchecked(trans_base + j) * *state_mexp.get_unchecked(j);
                }

                *beta.get_unchecked_mut(curr_base + y) = sum * scale;
            }
        }
    }

    // Gold score
    let mut gold_score = 0.0f64;
    for (t, &label) in gold_labels.iter().enumerate() {
        let base = t * num_labels;
        gold_score += exp_state[base + label as usize].ln();
    }
    for t in 1..n {
        let trans_idx =
            num_state_features + gold_labels[t - 1] as usize * num_labels + gold_labels[t] as usize;
        gold_score += params[trans_idx];
    }

    // State marginals
    unsafe {
        for t in 0..n {
            let base = t * num_labels;
            let inv_scale = 1.0 / *scale_factors.get_unchecked(t);

            for i in 0..chunks {
                let y = i * 4;
                let a0 = *alpha.get_unchecked(base + y);
                let a1 = *alpha.get_unchecked(base + y + 1);
                let a2 = *alpha.get_unchecked(base + y + 2);
                let a3 = *alpha.get_unchecked(base + y + 3);

                let b0 = *beta.get_unchecked(base + y);
                let b1 = *beta.get_unchecked(base + y + 1);
                let b2 = *beta.get_unchecked(base + y + 2);
                let b3 = *beta.get_unchecked(base + y + 3);

                *state_mexp.get_unchecked_mut(base + y) = a0 * b0 * inv_scale;
                *state_mexp.get_unchecked_mut(base + y + 1) = a1 * b1 * inv_scale;
                *state_mexp.get_unchecked_mut(base + y + 2) = a2 * b2 * inv_scale;
                *state_mexp.get_unchecked_mut(base + y + 3) = a3 * b3 * inv_scale;
            }
            for y in (chunks * 4)..num_labels {
                *state_mexp.get_unchecked_mut(base + y) =
                    *alpha.get_unchecked(base + y) * *beta.get_unchecked(base + y) * inv_scale;
            }
        }

        // Transition marginals
        for t in 0..n - 1 {
            let curr_base = t * num_labels;
            let next_base = (t + 1) * num_labels;

            for y_prev in 0..num_labels {
                let alpha_prev = *alpha.get_unchecked(curr_base + y_prev);
                let trans_row = y_prev * num_labels;

                for i in 0..chunks {
                    let y = i * 4;

                    let t0 = *exp_trans.get_unchecked(trans_row + y);
                    let t1 = *exp_trans.get_unchecked(trans_row + y + 1);
                    let t2 = *exp_trans.get_unchecked(trans_row + y + 2);
                    let t3 = *exp_trans.get_unchecked(trans_row + y + 3);

                    let s0 = *exp_state.get_unchecked(next_base + y);
                    let s1 = *exp_state.get_unchecked(next_base + y + 1);
                    let s2 = *exp_state.get_unchecked(next_base + y + 2);
                    let s3 = *exp_state.get_unchecked(next_base + y + 3);

                    let b0 = *beta.get_unchecked(next_base + y);
                    let b1 = *beta.get_unchecked(next_base + y + 1);
                    let b2 = *beta.get_unchecked(next_base + y + 2);
                    let b3 = *beta.get_unchecked(next_base + y + 3);

                    *trans_mexp.get_unchecked_mut(trans_row + y) += alpha_prev * t0 * s0 * b0;
                    *trans_mexp.get_unchecked_mut(trans_row + y + 1) += alpha_prev * t1 * s1 * b1;
                    *trans_mexp.get_unchecked_mut(trans_row + y + 2) += alpha_prev * t2 * s2 * b2;
                    *trans_mexp.get_unchecked_mut(trans_row + y + 3) += alpha_prev * t3 * s3 * b3;
                }
                for y_curr in (chunks * 4)..num_labels {
                    let prob = alpha_prev
                        * *exp_trans.get_unchecked(trans_row + y_curr)
                        * *exp_state.get_unchecked(next_base + y_curr)
                        * *beta.get_unchecked(next_base + y_curr);
                    *trans_mexp.get_unchecked_mut(trans_row + y_curr) += prob;
                }
            }
        }
    }

    // Gradient updates (common to both SIMD and non-SIMD paths)
    unsafe {
        for t in 0..n {
            let base = t * num_labels;
            for &attr_id in &attr_ids[t] {
                let attr_idx = attr_id as usize;
                if attr_idx < num_attrs {
                    let start = *attr_offsets.get_unchecked(attr_idx) as usize;
                    let end = *attr_offsets.get_unchecked(attr_idx + 1) as usize;
                    for i in start..end {
                        let (label_id, feature_id) = *attr_features_flat.get_unchecked(i);
                        *gradient.get_unchecked_mut(feature_id as usize) +=
                            *state_mexp.get_unchecked(base + label_id as usize);
                    }
                }
            }
        }

        // Transition gradient
        for y_prev in 0..num_labels {
            for y_curr in 0..num_labels {
                let trans_idx = num_state_features + y_prev * num_labels + y_curr;
                *gradient.get_unchecked_mut(trans_idx) +=
                    *trans_mexp.get_unchecked(y_prev * num_labels + y_curr);
            }
        }
    }

    log_norm - gold_score
}
