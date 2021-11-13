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
            vec!["A".to_string(), "b".to_string()],
            vec!["a".to_string(), "BC".to_string()],
        ];
        let output = super::featurizer(input);
        let expected: Vec<Vec<String>> = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "bc".to_string()],
        ];
        assert_eq!(output, expected);
    }
}
