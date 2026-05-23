extern crate underthesea_core;

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use underthesea_core::featurizers::CRFFeaturizer;

    #[test]
    fn test_crf_featurizer() {
        let sentences = vec![vec![
            vec!["Messi".to_string(), "X".to_string()],
            vec!["giành".to_string(), "X".to_string()],
            vec!["quả".to_string(), "X".to_string()],
            vec!["Bóng".to_string(), "X".to_string()],
            vec!["Đá".to_string(), "X".to_string()],
            vec!["1".to_string(), "X".to_string()],
        ]];
        let feature_configs = vec!["T[0]".to_string(), "T[0].is_in_dict".to_string()];
        let mut dictionary = HashSet::new();
        dictionary.insert("giành".to_string());
        dictionary.insert("quả".to_string());
        dictionary.insert("bóng".to_string());
        let new_featurizer = CRFFeaturizer::new(feature_configs, dictionary);
        assert_eq!(new_featurizer.feature_configs[0], "T[0]");
        let expected: Vec<Vec<Vec<String>>> = vec![vec![
            vec![
                "T[0]=Messi".to_string(),
                "T[0].is_in_dict=False".to_string(),
            ],
            vec!["T[0]=giành".to_string(), "T[0].is_in_dict=True".to_string()],
            vec!["T[0]=quả".to_string(), "T[0].is_in_dict=True".to_string()],
            vec!["T[0]=Bóng".to_string(), "T[0].is_in_dict=True".to_string()],
            vec!["T[0]=Đá".to_string(), "T[0].is_in_dict=False".to_string()],
            vec!["T[0]=1".to_string(), "T[0].is_in_dict=False".to_string()],
        ]];
        let output = new_featurizer.process(sentences);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_crf_featurizer_column_index() {
        let sentences = vec![vec![
            vec!["sinh".to_string(), "A".to_string()],
            vec!["viên".to_string(), "B".to_string()],
            vec!["đi".to_string(), "C".to_string()],
            vec!["học".to_string(), "D".to_string()],
        ]];
        let feature_configs = vec!["T[0][0]".to_string(), "T[0][1]".to_string()];
        let new_featurizer = CRFFeaturizer::new(feature_configs, HashSet::new());
        let expected: Vec<Vec<Vec<String>>> = vec![vec![
            vec!["T[0][0]=sinh".to_string(), "T[0][1]=A".to_string()],
            vec!["T[0][0]=viên".to_string(), "T[0][1]=B".to_string()],
            vec!["T[0][0]=đi".to_string(), "T[0][1]=C".to_string()],
            vec!["T[0][0]=học".to_string(), "T[0][1]=D".to_string()],
        ]];
        let output = new_featurizer.process(sentences);
        assert_eq!(output, expected);
    }
}
