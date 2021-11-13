extern crate regex;
extern crate pyo3;

use pyo3::prelude::*;
mod featurizers;

#[pyfunction]
fn featurizer(sentences: Vec<Vec<Vec<String>>>, features: Vec<String>) -> PyResult<Vec<Vec<Vec<String>>>> {
    let output = featurizers::featurizer(sentences, features);
    Ok(output)
}

#[pymodule]
fn underthesea_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(featurizer, m)?)?;
    Ok(())
}