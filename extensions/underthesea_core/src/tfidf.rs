//! tfidf.rs
//! 
//! Provides functionality for computing Term Frequency-Inverse Document Frequency (TFIDF) vectors.
//! 
//! Author: Vu Anh 
//! Date: 2023-07-29

use std::collections::{HashMap, HashSet};

pub struct TfidfFeaturizer {
    idf: Vec<f64>,
    term_to_index: HashMap<String, usize>
}

impl TfidfFeaturizer {
    pub fn new() -> Self {
        TfidfFeaturizer {
            idf: Vec::new(),
            term_to_index: HashMap::new()
        }
    }

    pub fn get_idf(&self) -> &Vec<f64> {
        &self.idf
    }

    fn compute_idf(&mut self, documents: &[Vec<String>]) {
        let n = documents.len() as f64;

        let mut word_freq = HashMap::new();

        for doc in documents.iter() {
            let mut seen_terms = HashSet::new();

            for term in doc {
                if !seen_terms.contains(term) {
                    let idx = match self.term_to_index.get(term) {
                        Some(&existing_idx) => existing_idx,
                        None => {
                            let new_idx = self.term_to_index.len();
                            self.term_to_index.insert(term.clone(), new_idx);
                            new_idx
                        }
                    };
                    *word_freq.entry(idx).or_insert(0.0) += 1.0;
                    seen_terms.insert(term.clone());
                }
            }
        }

        self.idf.resize(self.term_to_index.len(), 0.0);
        for(&idx, &freq) in &word_freq {
            self.idf[idx] = (n / freq).ln();
        }
    }

    pub fn train(&mut self, texts: &[&str]){
        let documents: Vec<Vec<String>> = texts.iter().map(|text| {
            text.split_whitespace().map(|word| word.to_string()).collect()
        }).collect();

        self.compute_idf(&documents);
    }

    pub fn predict(&self, texts: &Vec<&str>) -> Vec<Vec<f64>> {
        texts.iter().map(|text| {
            let words: Vec<String> = text.split_whitespace().map(|word| word.to_string()).collect();
            let mut tfidf_vector = vec![0.0; self.term_to_index.len()];

            // compute term frequence for this text
            let mut tf = HashMap::new();
            for word in &words {
                *tf.entry(word).or_insert(0.0) += 1.0;
            }

            let keys: Vec<_> = tf.keys().cloned().collect();
            for word in keys {
                if let Some(freq) = tf.get_mut(&word) {
                    *freq /= words.len() as f64;
                }
            }

            // compute tfidf values
            for (word, &index) in &self.term_to_index {
                if let Some(&term_freq) = tf.get(word) {
                    tfidf_vector[index] = term_freq * self.idf[index];
                }
            }

            tfidf_vector
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::cosine_similarity;

    #[test]
    fn test_constructor(){
        TfidfFeaturizer::new();
    }

    #[test]
    fn test_train_tfidf() {
        let mut tfidf_featurizer = TfidfFeaturizer::new();
        let texts = vec![
            "i love you",
            "you hate me",
            "me too"
        ];

        // Train tfidf vectorizer
        tfidf_featurizer.train(&texts);

        // vocab: i love you hate me too
        
        let idf_actual = tfidf_featurizer.get_idf();
        assert_eq!(idf_actual.len(), 6);

        let idf_expected = vec![
            (3.0f64/1.0f64).ln(),
            (3.0f64/1.0f64).ln(),
            (3.0f64/2.0f64).ln(),
            (3.0f64/1.0f64).ln(),
            (3.0f64/2.0f64).ln(),
            (3.0f64/1.0f64).ln(),
        ];
        assert!((cosine_similarity(&idf_actual, &idf_expected) - 1.0).abs() < 1e-9);
        
        // Predict tfidf values
        let output = tfidf_featurizer.predict(&texts);
        assert!(output.len() == 3);

        // Document 1: "i love you"
        let doc1_actual = output[0].clone();
        let doc1_expected = vec![
            (1.0f64 / 3.0f64) * (3.0f64).ln() ,
            (1.0f64 / 3.0f64) * (3.0f64).ln() ,
            (1.0f64 / 3.0f64) * (3.0f64 / 2.0f64).ln() ,
            0.0f64,
            0.0f64,
            0.0f64
        ];
        assert!((cosine_similarity(&doc1_actual, &doc1_expected) - 1.0).abs() < 1e-9);

        // Document 2: "you hate me"
        let doc2_actual = output[1].clone();
        let doc2_expected = vec![
            0.0f64,
            0.0f64,
            (1.0f64 / 3.0f64) * (3.0f64 / 2.0f64).ln(),
            (1.0f64 / 3.0f64) * (3.0f64 / 1.0f64).ln(),
            (1.0f64 / 3.0f64) * (3.0f64 / 2.0f64).ln(),
            0.0f64
        ];
        assert!((cosine_similarity(&doc2_actual, &doc2_expected) - 1.0).abs() < 1e-9);
    }
}