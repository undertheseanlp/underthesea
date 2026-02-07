pub mod args;
pub mod dictionary;
pub mod hash;
pub mod inference;
pub mod matrix;

use args::{check_header, Args, LossName};
use dictionary::Dictionary;
use inference::HSTree;
use matrix::FastTextMatrix;

use std::fs::File;
use std::io::{self, BufReader, Read};

/// A loaded FastText model ready for inference.
pub struct FastTextModel {
    args: Args,
    dictionary: Dictionary,
    input_matrix: FastTextMatrix,
    output_matrix: FastTextMatrix,
    hs_tree: Option<HSTree>,
}

impl FastTextModel {
    /// Load a FastText model from a `.bin` (dense) or `.ftz` (quantized) file.
    pub fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let quant = path.ends_with(".ftz");

        check_header(&mut reader)?;
        let args = Args::load(&mut reader)?;
        let dictionary = Dictionary::load(&mut reader, &args)?;

        let file_quant = read_bool(&mut reader)?;
        let input_matrix = FastTextMatrix::load(&mut reader, quant && file_quant)?;

        let output_quant = read_bool(&mut reader)?;
        let output_matrix = FastTextMatrix::load(&mut reader, quant && output_quant)?;

        let hs_tree = if args.loss == LossName::HierarchicalSoftmax {
            let counts = dictionary.get_label_counts();
            Some(HSTree::build(&counts))
        } else {
            None
        };

        Ok(FastTextModel {
            args,
            dictionary,
            input_matrix,
            output_matrix,
            hs_tree,
        })
    }

    /// Predict the top-k labels for the given text.
    pub fn predict(&self, text: &str, k: usize) -> Vec<(String, f32)> {
        inference::predict(
            text,
            k,
            &self.input_matrix,
            &self.output_matrix,
            &self.dictionary,
            self.args.dim as usize,
            self.args.loss,
            self.hs_tree.as_ref(),
        )
    }

    /// Get all label strings (without the `__label__` prefix).
    pub fn get_labels(&self) -> Vec<String> {
        self.dictionary
            .get_labels()
            .into_iter()
            .map(|l| l.strip_prefix("__label__").unwrap_or(&l).to_string())
            .collect()
    }

    /// Get model dimensionality.
    pub fn dim(&self) -> i32 {
        self.args.dim
    }

    /// Get number of words in the dictionary.
    pub fn nwords(&self) -> i32 {
        self.dictionary.nwords()
    }

    /// Get number of labels.
    pub fn nlabels(&self) -> i32 {
        self.dictionary.nlabels()
    }

    /// Get the hidden vector (averaged input embeddings) for a text.
    pub fn get_hidden(&self, text: &str) -> Vec<f32> {
        use matrix::Matrix;
        let features = self.dictionary.get_line_features(text);
        let dim = self.args.dim as usize;
        let mut hidden = vec![0.0f32; dim];
        for &feat_id in &features {
            if (feat_id as usize) < self.input_matrix.rows() {
                self.input_matrix.add_row_to(feat_id as usize, &mut hidden);
            }
        }
        if !features.is_empty() {
            let scale = 1.0 / features.len() as f32;
            for h in hidden.iter_mut() {
                *h *= scale;
            }
        }
        hidden
    }

    /// Get the input feature IDs for a text.
    pub fn get_features(&self, text: &str) -> Vec<i32> {
        self.dictionary.get_line_features(text)
    }
}

fn read_bool<R: Read>(reader: &mut R) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}
