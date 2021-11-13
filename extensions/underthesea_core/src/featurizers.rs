
pub fn featurizer(sentences: Vec<String>) -> Vec<String> {
    return sentences.iter().map(|s| s.to_lowercase()).collect();
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_featurizer() {
        let input: Vec<String> = ["A", "b"].iter().map(|&s| s.into()).collect();
        let output = super::featurizer(input);
        let expected: Vec<String> = ["a", "b"].iter().map(|&s| s.into()).collect();
        assert_eq!(output, expected);
    }
}

