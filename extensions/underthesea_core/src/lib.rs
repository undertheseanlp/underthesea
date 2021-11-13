use pyo3::prelude::*;
mod featurizers;

#[pyfunction]
fn featurizer(input: Vec<Vec<String>>) -> PyResult<Vec<Vec<String>>> {
    let output = featurizers::featurizer(input);
    Ok(output)
}

#[pymodule]
fn underthesea_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(featurizer, m)?)?;
    Ok(())
}