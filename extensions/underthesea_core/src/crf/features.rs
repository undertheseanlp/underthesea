//! Feature functions for CRF models.
//!
//! This module defines the feature function types supported by the CRF:
//! - Unigram features: Single token features (T[0], T[-1])
//! - Bigram features: Adjacent token pairs (T[-1,0])
//! - Character trigram features: Character-level n-grams
//! - Transition features: Label-to-label transitions

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Types of feature functions supported by the CRF.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// State/emission feature: maps (attribute, label) to weight
    /// Example: "word=hello" + label "NOUN"
    State,

    /// Transition feature: maps (prev_label, curr_label) to weight
    /// Example: "NOUN" -> "VERB" transition
    Transition,

    /// Unigram feature: single token at a position
    /// Example: T[0], T[-1]
    Unigram,

    /// Bigram feature: adjacent token pairs
    /// Example: T[-1,0]
    Bigram,

    /// Character trigram feature: 3-character substrings
    /// Example: "hel", "ell", "llo" for "hello"
    CharacterTrigram,

    /// Custom user-defined feature type
    Custom(String),
}

/// A feature function that maps observations to feature values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFunction {
    /// Unique identifier for this feature
    pub id: u32,

    /// Type of feature
    pub feature_type: FeatureType,

    /// Source attribute ID (for state features)
    /// This is the attribute that triggers the feature
    pub source: u32,

    /// Target label ID (for state features) or source label (for transitions)
    pub target: u32,

    /// Weight of this feature
    pub weight: f64,
}

impl FeatureFunction {
    /// Create a new state feature.
    pub fn new_state(id: u32, attr_id: u32, label_id: u32, weight: f64) -> Self {
        Self {
            id,
            feature_type: FeatureType::State,
            source: attr_id,
            target: label_id,
            weight,
        }
    }

    /// Create a new transition feature.
    pub fn new_transition(id: u32, from_label: u32, to_label: u32, weight: f64) -> Self {
        Self {
            id,
            feature_type: FeatureType::Transition,
            source: from_label,
            target: to_label,
            weight,
        }
    }
}

/// Manages the mapping between string attributes and numeric IDs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributeIndex {
    /// Maps attribute strings to their IDs
    attr_to_id: HashMap<String, u32>,

    /// Maps IDs back to attribute strings
    id_to_attr: Vec<String>,
}

impl AttributeIndex {
    /// Create a new empty attribute index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create an ID for the given attribute.
    pub fn get_or_insert(&mut self, attr: &str) -> u32 {
        if let Some(&id) = self.attr_to_id.get(attr) {
            id
        } else {
            let id = self.id_to_attr.len() as u32;
            self.attr_to_id.insert(attr.to_string(), id);
            self.id_to_attr.push(attr.to_string());
            id
        }
    }

    /// Get the ID for an attribute, if it exists.
    pub fn get(&self, attr: &str) -> Option<u32> {
        self.attr_to_id.get(attr).copied()
    }

    /// Get the attribute for an ID, if it exists.
    pub fn get_attr(&self, id: u32) -> Option<&str> {
        self.id_to_attr.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of attributes.
    pub fn len(&self) -> usize {
        self.id_to_attr.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_attr.is_empty()
    }

    /// Iterate over all (id, attribute) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.id_to_attr
            .iter()
            .enumerate()
            .map(|(id, attr)| (id as u32, attr.as_str()))
    }
}

/// Manages the mapping between label strings and numeric IDs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabelIndex {
    /// Maps label strings to their IDs
    label_to_id: HashMap<String, u32>,

    /// Maps IDs back to label strings
    id_to_label: Vec<String>,
}

impl LabelIndex {
    /// Create a new empty label index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create an ID for the given label.
    pub fn get_or_insert(&mut self, label: &str) -> u32 {
        if let Some(&id) = self.label_to_id.get(label) {
            id
        } else {
            let id = self.id_to_label.len() as u32;
            self.label_to_id.insert(label.to_string(), id);
            self.id_to_label.push(label.to_string());
            id
        }
    }

    /// Get the ID for a label, if it exists.
    pub fn get(&self, label: &str) -> Option<u32> {
        self.label_to_id.get(label).copied()
    }

    /// Get the label for an ID, if it exists.
    pub fn get_label(&self, id: u32) -> Option<&str> {
        self.id_to_label.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of labels.
    pub fn len(&self) -> usize {
        self.id_to_label.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_label.is_empty()
    }

    /// Get all labels as a slice.
    pub fn labels(&self) -> &[String] {
        &self.id_to_label
    }

    /// Iterate over all (id, label) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.id_to_label
            .iter()
            .enumerate()
            .map(|(id, label)| (id as u32, label.as_str()))
    }
}

/// Extract character trigrams from a string.
pub fn extract_char_trigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        return vec![text.to_string()];
    }

    chars
        .windows(3)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_index() {
        let mut index = AttributeIndex::new();
        let id1 = index.get_or_insert("word=hello");
        let id2 = index.get_or_insert("word=world");
        let id3 = index.get_or_insert("word=hello");

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3); // Same attribute should get same ID
        assert_eq!(index.len(), 2);
        assert_eq!(index.get("word=hello"), Some(0));
        assert_eq!(index.get_attr(0), Some("word=hello"));
    }

    #[test]
    fn test_label_index() {
        let mut index = LabelIndex::new();
        let id1 = index.get_or_insert("B-PER");
        let id2 = index.get_or_insert("I-PER");
        let id3 = index.get_or_insert("O");

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(index.len(), 3);
        assert_eq!(index.get("B-PER"), Some(0));
        assert_eq!(index.get_label(1), Some("I-PER"));
    }

    #[test]
    fn test_char_trigrams() {
        let trigrams = extract_char_trigrams("hello");
        assert_eq!(trigrams, vec!["hel", "ell", "llo"]);

        let short = extract_char_trigrams("ab");
        assert_eq!(short, vec!["ab"]);
    }

    #[test]
    fn test_feature_function() {
        let state = FeatureFunction::new_state(0, 1, 2, 0.5);
        assert_eq!(state.feature_type, FeatureType::State);
        assert_eq!(state.source, 1);
        assert_eq!(state.target, 2);
        assert_eq!(state.weight, 0.5);

        let trans = FeatureFunction::new_transition(1, 0, 1, -0.3);
        assert_eq!(trans.feature_type, FeatureType::Transition);
        assert_eq!(trans.source, 0);
        assert_eq!(trans.target, 1);
    }
}
