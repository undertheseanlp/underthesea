extern crate regex;
extern crate pyo3;

use pyo3::prelude::*;
use std::collections::HashSet;

mod featurizers;

#[pyfunction]
fn featurizer(sentences: Vec<Vec<Vec<String>>>, features: Vec<String>, dictionary: HashSet<String>) -> PyResult<Vec<Vec<Vec<String>>>> {
    let output = featurizers::featurizer(sentences, features, dictionary);
    Ok(output)
}

#[pymodule]
fn underthesea_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(featurizer, m)?)?;
    Ok(())
}