//! Text processing utilities for NLP.
//!
//! This module provides text vectorization and classification tools:
//! - `TfIdfVectorizer`: Transform text documents into TF-IDF feature vectors

pub mod tfidf;

pub use tfidf::{TfIdfConfig, TfIdfVectorizer};
