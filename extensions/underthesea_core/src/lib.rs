extern crate regex;
extern crate pyo3;

use pyo3::prelude::*;
use std::collections::HashSet;

pub mod featurizers;

#[pyclass]
pub struct CRFFeaturizer {
    pub object: featurizers::CRFFeaturizer
}


#[pymethods]
impl CRFFeaturizer {
    #[new]
    pub fn new(feature_configs: Vec<String>, dictionary: HashSet<String>) -> PyResult<Self> {
        Ok(CRFFeaturizer {
            object: featurizers::CRFFeaturizer::new(feature_configs, dictionary)
        })
    }

    pub fn process(self_: PyRef<Self>, sentences: Vec<Vec<Vec<String>>>) -> PyResult<Vec<Vec<Vec<String>>>> {
        let output = self_.object.process(sentences);
        Ok(output)
    }

    // pub fn process(self_: PyRef<Self>, sentences: Vec<Vec<Vec<String>>>) -> PyResult<String> {
    //     Ok("5".to_string())
    // }
}

#[pymodule]
fn underthesea_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CRFFeaturizer>()?;
    Ok(())
}