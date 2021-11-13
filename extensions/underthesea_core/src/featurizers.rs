pub fn featurizer(sentences: Vec<Vec<String>>) -> Vec<Vec<String>> {
    return sentences.iter().map(
        |item| item.iter().map(|s| s.to_lowercase()).collect()
    ).collect();
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_featurizer() {
        let input = vec![
            vec!["Cải cách".to_string(), "bóng".to_string(), "đá".to_string()],
            vec!["Việt".to_string(), "Nam".to_string()],
        ];
        let output = super::featurizer(input);
        let expected: Vec<Vec<String>> = vec![
            vec!["cải cách".to_string(), "bóng".to_string(), "đá".to_string()],
            vec!["việt".to_string(), "nam".to_string()],
        ];
        assert_eq!(output, expected);
    }
}
