//! Vietnamese Text Preprocessor
//!
//! Configurable text preprocessing pipeline for Vietnamese NLP.
//! Serializable with serde/bincode — gets packed into the same .bin file
//! as the TextClassifier model, so preprocessing config always travels
//! with the model.
//!
//! ## Python Usage
//!
//! ```python
//! # Default Vietnamese preprocessing
//! pp = TextPreprocessor()
//! pp.transform("sp ko đẹp lắm!!!")
//! # -> "sản phẩm không NEG_đẹp NEG_lắm!"
//!
//! # Custom teencode dictionary
//! pp = TextPreprocessor(teencode={"ko": "không", "dc": "được"})
//!
//! # Custom negation words + window
//! pp = TextPreprocessor(
//!     negation_words=["không", "chưa", "chẳng"],
//!     negation_window=3,
//! )
//!
//! # Disable specific steps
//! pp = TextPreprocessor(lowercase=False, remove_urls=False)
//!
//! # No teencode, no negation
//! pp = TextPreprocessor(teencode=None, negation_words=None)
//! ```

use hashbrown::HashMap;
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap as StdHashMap;

/// Default Vietnamese teencode dictionary.
/// Users can override this entirely by passing their own dict.
fn default_teencode() -> HashMap<String, String> {
    [
        ("ko", "không"),
        ("k", "không"),
        ("hok", "không"),
        ("hem", "không"),
        ("dc", "được"),
        ("đc", "được"),
        ("dk", "được"),
        ("ntn", "như thế nào"),
        ("nc", "nói chuyện"),
        ("nt", "nhắn tin"),
        ("cx", "cũng"),
        ("cg", "cũng"),
        ("vs", "với"),
        ("vl", "vãi"),
        ("bt", "bình thường"),
        ("bth", "bình thường"),
        ("lg", "lượng"),
        ("tl", "trả lời"),
        ("ms", "mới"),
        ("r", "rồi"),
        ("mn", "mọi người"),
        ("mk", "mình"),
        ("ok", "tốt"),
        ("oke", "tốt"),
        ("sp", "sản phẩm"),
        ("hqua", "hôm qua"),
        ("hnay", "hôm nay"),
        ("tks", "cảm ơn"),
        ("thanks", "cảm ơn"),
        ("thank", "cảm ơn"),
        ("j", "gì"),
        ("z", "vậy"),
        ("v", "vậy"),
        ("đt", "điện thoại"),
        ("dt", "điện thoại"),
        ("lm", "làm"),
        ("ns", "nói"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Default Vietnamese negation words.
/// Users can override this entirely by passing their own list.
fn default_negation_words() -> Vec<String> {
    [
        "không", "chẳng", "chả", "chưa", "đừng", "ko", "hok", "hem", "chăng",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Configurable Vietnamese text preprocessor.
///
/// Pipeline steps (applied in order):
///   1. Unicode NFC normalization
///   2. Lowercase
///   3. URL removal
///   4. Repeated character normalization  ("đẹppp" → "đẹpp")
///   5. Punctuation normalization         ("!!!" → "!", "????" → "?")
///   6. Teencode expansion                ("ko" → "không", "dc" → "được")
///   7. Negation marking                  ("không tốt" → "không NEG_tốt")
///
/// All fields are serializable. When embedded in TextClassifier and saved
/// to .bin, the entire config is persisted — no separate config file needed.
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct TextPreprocessor {
    /// Convert text to lowercase. Default: true
    pub lowercase: bool,
    /// Apply Unicode NFC normalization. Default: true
    pub unicode_normalize: bool,
    /// Remove URLs (http/https/www). Default: true
    pub remove_urls: bool,
    /// Reduce 3+ repeated chars to 2 ("đẹppp" → "đẹpp"). Default: true
    pub normalize_repeated_chars: bool,
    /// Reduce repeated punctuation ("!!!" → "!"). Default: true
    pub normalize_punctuation: bool,
    /// Teencode expansion dictionary. None = disabled.
    /// Key: teencode word, Value: replacement.
    /// Default: Vietnamese teencode (ko→không, dc→được, ...)
    pub teencode: Option<HashMap<String, String>>,
    /// Negation words that trigger NEG_ marking. None = disabled.
    /// Default: ["không", "chẳng", "chả", "chưa", "đừng", ...]
    pub negation_words: Option<Vec<String>>,
    /// Number of words after negation word to mark with NEG_ prefix.
    /// Default: 2
    pub negation_window: usize,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            lowercase: true,
            unicode_normalize: true,
            remove_urls: true,
            normalize_repeated_chars: true,
            normalize_punctuation: true,
            teencode: Some(default_teencode()),
            negation_words: Some(default_negation_words()),
            negation_window: 2,
        }
    }
}

impl TextPreprocessor {
    /// Apply all enabled preprocessing steps to a single text.
    pub fn transform(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Step 1: Unicode NFC normalization
        if self.unicode_normalize {
            use unicode_normalization::UnicodeNormalization;
            result = result.nfc().collect();
        }

        // Step 2: Lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // Step 3: URL removal
        if self.remove_urls {
            let url_re = Regex::new(r"https?://\S+|www\.\S+").unwrap();
            result = url_re.replace_all(&result, " ").to_string();
        }

        // Step 4: Repeated character normalization (no regex backreference needed)
        // Skip punctuation chars — those are handled by step 5
        if self.normalize_repeated_chars {
            let mut chars: Vec<char> = Vec::with_capacity(result.len());
            let mut count = 0u32;
            let mut prev: Option<char> = None;
            for c in result.chars() {
                if Some(c) == prev && !c.is_ascii_punctuation() {
                    count += 1;
                    if count < 2 {
                        chars.push(c);
                    }
                } else {
                    chars.push(c);
                    prev = Some(c);
                    count = 0;
                }
            }
            result = chars.into_iter().collect();
        }

        // Step 5: Punctuation normalization
        if self.normalize_punctuation {
            let excl_re = Regex::new(r"!{2,}").unwrap();
            let ques_re = Regex::new(r"\?{2,}").unwrap();
            let dots_re = Regex::new(r"\.{4,}").unwrap();
            result = excl_re.replace_all(&result, "!").to_string();
            result = ques_re.replace_all(&result, "?").to_string();
            result = dots_re.replace_all(&result, "...").to_string();
        }

        let words: Vec<String> = result.split_whitespace().map(|s| s.to_string()).collect();
        if words.is_empty() {
            return String::new();
        }

        // Step 6: Teencode expansion
        let expanded: Vec<String> = if let Some(ref tc) = self.teencode {
            words
                .iter()
                .map(|w| {
                    let stripped = w.trim_matches(|c: char| ".,!?;:".contains(c));
                    tc.get(stripped).cloned().unwrap_or_else(|| w.clone())
                })
                .collect()
        } else {
            words
        };

        // Step 7: Negation marking
        let final_words = if let Some(ref neg_words) = self.negation_words {
            let neg_set: std::collections::HashSet<&str> =
                neg_words.iter().map(|s| s.as_str()).collect();

            let mut marked = expanded.clone();
            for (i, w) in expanded.iter().enumerate() {
                let stripped = w.trim_matches(|c: char| ".,!?;:".contains(c));
                if neg_set.contains(stripped) {
                    let end = (i + 1 + self.negation_window).min(expanded.len());
                    for j in (i + 1)..end {
                        marked[j] = format!("NEG_{}", expanded[j]);
                    }
                }
            }
            marked
        } else {
            expanded
        };

        final_words.join(" ")
    }

    /// Apply preprocessing to a batch of texts.
    pub fn transform_batch(&self, texts: &[String]) -> Vec<String> {
        texts.iter().map(|t| self.transform(t)).collect()
    }
}

#[pymethods]
impl TextPreprocessor {
    /// Create a new TextPreprocessor.
    ///
    /// With no arguments, uses Vietnamese defaults for all steps.
    /// Pass custom dicts/lists to override, or None to disable.
    ///
    /// Examples (Python):
    /// ```python
    /// # Vietnamese defaults
    /// pp = TextPreprocessor()
    ///
    /// # Custom teencode only
    /// pp = TextPreprocessor(teencode={"ko": "không", "dc": "được"})
    ///
    /// # Custom negation
    /// pp = TextPreprocessor(negation_words=["không", "chưa"], negation_window=3)
    ///
    /// # Disable teencode and negation
    /// pp = TextPreprocessor(teencode=None, negation_words=None, use_defaults=False)
    /// ```
    #[new]
    #[pyo3(signature = (
        lowercase=true,
        unicode_normalize=true,
        remove_urls=true,
        normalize_repeated_chars=true,
        normalize_punctuation=true,
        teencode=None,
        negation_words=None,
        negation_window=2,
        use_defaults=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        lowercase: bool,
        unicode_normalize: bool,
        remove_urls: bool,
        normalize_repeated_chars: bool,
        normalize_punctuation: bool,
        teencode: Option<StdHashMap<String, String>>,
        negation_words: Option<Vec<String>>,
        negation_window: usize,
        use_defaults: bool,
    ) -> Self {
        // When use_defaults=True (default) and user didn't pass custom values,
        // fill in Vietnamese defaults. When use_defaults=False, treat None as disabled.
        let tc = match (teencode, use_defaults) {
            (Some(custom), _) => Some(custom.into_iter().collect()), // StdHashMap → hashbrown
            (None, true) => Some(default_teencode()), // no custom + defaults on → use defaults
            (None, false) => None,                    // no custom + defaults off → disabled
        };
        let nw = match (negation_words, use_defaults) {
            (Some(custom), _) => Some(custom),
            (None, true) => Some(default_negation_words()),
            (None, false) => None,
        };

        Self {
            lowercase,
            unicode_normalize,
            remove_urls,
            normalize_repeated_chars,
            normalize_punctuation,
            teencode: tc,
            negation_words: nw,
            negation_window,
        }
    }

    /// Preprocess a single text string.
    #[pyo3(name = "transform")]
    fn py_transform(&self, text: &str) -> String {
        self.transform(text)
    }

    /// Preprocess a list of texts.
    #[pyo3(name = "transform_batch")]
    fn py_transform_batch(&self, texts: Vec<String>) -> Vec<String> {
        self.transform_batch(&texts)
    }

    /// Get the teencode dictionary (or None if disabled).
    #[getter]
    fn get_teencode(&self) -> Option<StdHashMap<String, String>> {
        self.teencode
            .as_ref()
            .map(|tc| tc.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    /// Get the negation words list (or None if disabled).
    #[getter]
    fn get_negation_words(&self) -> Option<Vec<String>> {
        self.negation_words.clone()
    }

    /// Get the negation window size.
    #[getter]
    fn get_negation_window(&self) -> usize {
        self.negation_window
    }

    fn __repr__(&self) -> String {
        let mut steps = Vec::new();
        if self.unicode_normalize {
            steps.push("unicode_nfc".to_string());
        }
        if self.lowercase {
            steps.push("lowercase".to_string());
        }
        if self.remove_urls {
            steps.push("remove_urls".to_string());
        }
        if self.normalize_repeated_chars {
            steps.push("norm_repeated_chars".to_string());
        }
        if self.normalize_punctuation {
            steps.push("norm_punctuation".to_string());
        }
        if let Some(ref tc) = self.teencode {
            steps.push(format!("teencode({} rules)", tc.len()));
        }
        if let Some(ref nw) = self.negation_words {
            steps.push(format!(
                "negation({} words, window={})",
                nw.len(),
                self.negation_window
            ));
        }
        format!("TextPreprocessor([{}])", steps.join(" → "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_preprocessor() {
        let pp = TextPreprocessor::default();
        assert_eq!(pp.transform("Sản phẩm ko đẹp"), "sản phẩm không NEG_đẹp");
    }

    #[test]
    fn test_custom_teencode() {
        let mut tc = HashMap::new();
        tc.insert("abc".to_string(), "xyz".to_string());
        let pp = TextPreprocessor {
            teencode: Some(tc),
            negation_words: None,
            ..Default::default()
        };
        assert_eq!(pp.transform("abc test"), "xyz test");
    }

    #[test]
    fn test_custom_negation_window_3() {
        let pp = TextPreprocessor {
            negation_window: 3,
            ..Default::default()
        };
        let result = pp.transform("không tốt lắm đâu nhé");
        assert!(result.contains("NEG_tốt"));
        assert!(result.contains("NEG_lắm"));
        assert!(result.contains("NEG_đâu"));
        assert!(!result.contains("NEG_nhé"));
    }

    #[test]
    fn test_url_removal() {
        let pp = TextPreprocessor::default();
        let result = pp.transform("Check https://example.com ok");
        assert!(!result.contains("https"));
    }

    #[test]
    fn test_repeated_chars() {
        let pp = TextPreprocessor::default();
        assert_eq!(pp.transform("đẹppppp"), "đẹpp");
    }

    #[test]
    fn test_punctuation_normalization() {
        let pp = TextPreprocessor::default();
        assert_eq!(pp.transform("hay!!!"), "hay!");
        assert_eq!(pp.transform("sao????"), "sao?");
        assert_eq!(pp.transform("hmm....."), "hmm...");
    }

    #[test]
    fn test_teencode_disabled() {
        let pp = TextPreprocessor {
            teencode: None,
            ..Default::default()
        };
        let result = pp.transform("ko đẹp");
        assert!(result.contains("ko")); // not expanded
    }

    #[test]
    fn test_negation_disabled() {
        let pp = TextPreprocessor {
            negation_words: None,
            ..Default::default()
        };
        let result = pp.transform("không tốt");
        assert!(!result.contains("NEG_")); // no negation marking
    }

    #[test]
    fn test_all_disabled() {
        let pp = TextPreprocessor {
            lowercase: false,
            unicode_normalize: false,
            remove_urls: false,
            normalize_repeated_chars: false,
            normalize_punctuation: false,
            teencode: None,
            negation_words: None,
            negation_window: 2,
        };
        assert_eq!(pp.transform("Ko Đẹp!!!"), "Ko Đẹp!!!");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let pp = TextPreprocessor::default();
        let bytes = bincode::serialize(&pp).unwrap();
        let pp2: TextPreprocessor = bincode::deserialize(&bytes).unwrap();
        assert_eq!(pp.transform("ko đẹp"), pp2.transform("ko đẹp"));
    }

    #[test]
    fn test_custom_teencode_serialization() {
        let mut tc = HashMap::new();
        tc.insert("tks".to_string(), "thanks".to_string());
        let pp = TextPreprocessor {
            teencode: Some(tc),
            ..Default::default()
        };
        let bytes = bincode::serialize(&pp).unwrap();
        let pp2: TextPreprocessor = bincode::deserialize(&bytes).unwrap();
        assert_eq!(pp2.teencode.as_ref().unwrap().len(), 1);
        assert_eq!(pp.transform("tks"), pp2.transform("tks"));
    }
}
