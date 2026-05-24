#[cfg(test)]
mod tests {
    use underthesea_core::crf::serialization::{CRFFormat, ModelLoader};

    #[test]
    fn test_load_crfsuite_model() {
        let model = ModelLoader::new()
            .load("tests/wt_crf_2018_09_13.bin", CRFFormat::Auto)
            .unwrap();
        assert!(model.num_labels > 0);
        assert!(model.num_attributes > 0);
    }
}
