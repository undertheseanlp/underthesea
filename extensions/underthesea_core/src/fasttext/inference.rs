use super::args::LossName;
use super::dictionary::Dictionary;
use super::matrix::{FastTextMatrix, Matrix};
use std::collections::BinaryHeap;

/// Wrapper for f32 that implements Ord for use in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// A node in the Huffman tree for hierarchical softmax.
#[derive(Debug, Clone)]
struct HSNode {
    parent: i32,
    left: i32,
    right: i32,
    count: i64,
    #[allow(dead_code)]
    binary: bool,
}

/// Hierarchical softmax tree built from label counts.
pub struct HSTree {
    tree: Vec<HSNode>,
    osz: usize,
}

impl HSTree {
    /// Build a Huffman tree from label counts, matching FastText C++ implementation.
    pub fn build(counts: &[i64]) -> Self {
        let osz = counts.len();
        let tree_size = 2 * osz - 1;
        let mut tree: Vec<HSNode> = (0..tree_size)
            .map(|_| HSNode {
                parent: -1,
                left: -1,
                right: -1,
                count: 1_000_000_000_000_000,
                binary: false,
            })
            .collect();

        for i in 0..osz {
            tree[i].count = counts[i];
        }

        let mut leaf = osz as i32 - 1;
        let mut node = osz;
        for i in osz..tree_size {
            let mut mini = [0i32; 2];
            for j in 0..2 {
                if leaf >= 0 && tree[leaf as usize].count < tree[node].count {
                    mini[j] = leaf;
                    leaf -= 1;
                } else {
                    mini[j] = node as i32;
                    node += 1;
                }
            }
            tree[i].left = mini[0];
            tree[i].right = mini[1];
            tree[i].count = tree[mini[0] as usize].count + tree[mini[1] as usize].count;
            tree[mini[0] as usize].parent = i as i32;
            tree[mini[1] as usize].parent = i as i32;
            tree[mini[1] as usize].binary = true;
        }

        HSTree { tree, osz }
    }

    /// DFS through the tree to find top-k labels.
    /// Uses log-space scores matching C++ FastText v0.9.2 (loss.cc).
    fn dfs(
        &self,
        k: usize,
        threshold: f32,
        node: usize,
        score: f32,
        hidden: &[f32],
        output: &FastTextMatrix,
        heap: &mut BinaryHeap<std::cmp::Reverse<(FloatOrd, usize)>>,
    ) {
        if score < std_log(threshold) {
            return;
        }
        if heap.len() == k && score < heap.peek().unwrap().0 .0 .0 {
            return;
        }

        let n = &self.tree[node];
        if n.left == -1 && n.right == -1 {
            heap.push(std::cmp::Reverse((FloatOrd(score), node)));
            if heap.len() > k {
                heap.pop();
            }
            return;
        }

        let f = output.dot_row(hidden, node - self.osz);
        let f = sigmoid(f);

        // C++ FastText v0.9.2: left gets std_log(1-f), right gets std_log(f)
        self.dfs(
            k,
            threshold,
            n.left as usize,
            score + std_log(1.0 - f),
            hidden,
            output,
            heap,
        );
        self.dfs(
            k,
            threshold,
            n.right as usize,
            score + std_log(f),
            hidden,
            output,
            heap,
        );
    }
}

/// Predict the top-k labels for the given text.
pub fn predict(
    text: &str,
    k: usize,
    input_matrix: &FastTextMatrix,
    output_matrix: &FastTextMatrix,
    dictionary: &Dictionary,
    dim: usize,
    loss: LossName,
    hs_tree: Option<&HSTree>,
) -> Vec<(String, f32)> {
    let features = dictionary.get_line_features(text);

    if features.is_empty() {
        return Vec::new();
    }

    // Average input embeddings -> hidden vector
    let mut hidden = vec![0.0f32; dim];
    for &feat_id in &features {
        if (feat_id as usize) < input_matrix.rows() {
            input_matrix.add_row_to(feat_id as usize, &mut hidden);
        }
    }
    let scale = 1.0 / features.len() as f32;
    for h in hidden.iter_mut() {
        *h *= scale;
    }

    let nlabels = dictionary.nlabels() as usize;
    let k = k.min(nlabels);

    let label_scores = match loss {
        LossName::HierarchicalSoftmax => {
            predict_hs(k, &hidden, output_matrix, hs_tree.unwrap(), nlabels)
        }
        _ => predict_softmax(k, &hidden, output_matrix, nlabels),
    };

    label_scores
        .iter()
        .map(|&(score, label_idx)| {
            let label = dictionary.get_label(label_idx as i32);
            let label = label.strip_prefix("__label__").unwrap_or(label).to_string();
            (label, score)
        })
        .collect()
}

/// Hierarchical softmax prediction using DFS tree traversal.
fn predict_hs(
    k: usize,
    hidden: &[f32],
    output: &FastTextMatrix,
    tree: &HSTree,
    _nlabels: usize,
) -> Vec<(f32, usize)> {
    let mut heap: BinaryHeap<std::cmp::Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
    let root = 2 * tree.osz - 2;
    tree.dfs(k, 0.0, root, 0.0, hidden, output, &mut heap);

    let mut results: Vec<(f32, usize)> = heap
        .into_iter()
        .map(|std::cmp::Reverse((FloatOrd(score), idx))| (score.exp(), idx))
        .collect();
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Standard softmax prediction.
fn predict_softmax(
    k: usize,
    hidden: &[f32],
    output: &FastTextMatrix,
    nlabels: usize,
) -> Vec<(f32, usize)> {
    let mut logits = vec![0.0f32; nlabels];
    for i in 0..nlabels {
        logits[i] = output.dot_row(hidden, i);
    }
    softmax(&mut logits);

    let mut indices: Vec<usize> = (0..nlabels).collect();
    indices.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    indices.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indices.iter().map(|&i| (logits[i], i)).collect()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Matches C++ FastText std_log: log(x + 1e-5) to avoid log(0).
#[inline]
fn std_log(x: f32) -> f32 {
    (x + 1e-5).ln()
}

fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(logits[0] < logits[1]);
        assert!(logits[1] < logits[2]);
    }

    #[test]
    fn test_softmax_equal() {
        let mut logits = vec![1.0, 1.0, 1.0];
        softmax(&mut logits);
        for &v in &logits {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_hs_tree_build() {
        let counts = vec![10, 5, 1];
        let tree = HSTree::build(&counts);
        assert_eq!(tree.osz, 3);
        assert_eq!(tree.tree.len(), 5);
        assert!(tree.tree[4].left >= 0);
        assert!(tree.tree[4].right >= 0);
    }
}
