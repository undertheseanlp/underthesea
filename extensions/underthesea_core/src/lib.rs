use pyo3::prelude::*;

#[pyfunction]
fn sum_as_string(a: usize, b:usize) -> PyResult<String>{
    Ok((a + b).to_string())
}

#[pymodule]
fn underthesea_core(_py:Python, m:&PyModule) -> PyResult<()>{
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    Ok(())
}