use pyo3::prelude::*;
mod featurizers;
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn lower(a: String) -> PyResult<String> {
    let output = featurizers::lower(a);
    Ok(output)
}

#[pymodule]
fn underthesea_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(lower, m)?)?;
    Ok(())
}