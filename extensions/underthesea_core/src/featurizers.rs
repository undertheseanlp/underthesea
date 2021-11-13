use regex::{Regex};

pub struct FeatureTemplate {
    index1: i32,
    index2: Option<i32>,
    column: usize,
    function: Option<String>,
}

/// generate features for each token <position>-th in sentence
/// Sentence Example
/// Messi   X
/// giành   X
/// quả     X
pub fn generate_token_features(sentence: &Vec<Vec<String>>, position: usize, feature_templates: &Vec<FeatureTemplate>) -> Vec<String> {
    let mut features = Vec::new();
    for feature_template in feature_templates {
        let index1 = feature_template.index1;
        let column = feature_template.column;
        let index: i32 = position as i32 + index1;
        let n = sentence.len() as i32;
        let mut text: String = "".to_string();
        if index < 0 {
            features.push("BOS".to_string());
        } else if index >= n {
            features.push("EOS".to_string());
        }
        else {
            match sentence.get(index as usize) {
                None => {}
                Some(s) => {
                    match s.get(column) {
                        None => {}
                        Some(current_text) => {
                            text = current_text.to_string();

                        }
                    }
                }
            }

            // apply function
            match feature_template.function.as_ref() {
                None => {}
                Some(function_name) => {
                    match function_name.as_ref() {
                        "lower" => {
                            text = text.to_lowercase();
                        }
                        _ => {}
                    }
                }
            }
            features.push(text.to_string());
        }


    }
    return features;
}


pub fn featurizer(sentences: Vec<Vec<Vec<String>>>, feature_configs: Vec<String>) -> Vec<Vec<Vec<String>>> {
    // Step 1: Parse FeatureTemplates
    let re = Regex::new(
        r"T\[(?P<index1>-?\d+)(,(?P<index2>-?\d+))?](\[(?P<column>.*)])?(\.(?P<function>.*))?"
    ).unwrap();
    let mut feature_templates: Vec<FeatureTemplate> = Vec::new();
    for feature_config in feature_configs {
        let mut feature_template = FeatureTemplate {
            index1: 0,
            index2: None,
            column: 0,
            function: None,
        };

        for cap in re.captures_iter(feature_config.as_str()) {
            match cap.name("index1") {
                Some(s) => {
                    feature_template.index1 = s.as_str().parse::<i32>().unwrap();
                }
                _ => ()
            }
            match cap.name("index2") {
                Some(s) => {
                    feature_template.index2 = Option::from(s.as_str().parse::<i32>().unwrap());
                }
                _ => ()
            }

            // match cap.name("column") {
            //     Some(s) => {
            //         feature_template.column = s.as_str().parse::<i32>().unwrap();
            //     }
            //     _ => ()
            // }

            match cap.name("function") {
                Some(s) => {
                    feature_template.function = Option::from(s.as_str().parse::<String>().unwrap());
                }
                _ => ()
            }
        }

        feature_templates.push(feature_template);
    }

    // Step 2: Generate features
    let mut sentences_features = Vec::new();
    for sentence in sentences {
        // generate features for each sentence
        let mut sentence_features = Vec::new();
        for position in 0..sentence.len() {
            let token_features = generate_token_features(&sentence, position, &feature_templates);
            sentence_features.push(token_features);
        }
        sentences_features.push(sentence_features);
    }
    return sentences_features;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_featurizer() {
        let sentences = vec![
            vec![
                vec!["Messi".to_string(), "X".to_string()],
                vec!["giành".to_string(), "X".to_string()],
                vec!["quả".to_string(), "X".to_string()],
            ],
            vec![
                vec!["Bóng".to_string(), "X".to_string()],
                vec!["Vàng".to_string(), "X".to_string()]
            ]
        ];
        let features = vec![
            "T[-3].lower".to_string(),
            "T[-2].lower".to_string(),
            "T[-1].lower".to_string(),
            "T[0].lower".to_string(),
            "T[1].lower".to_string(),
            "T[2].lower".to_string(),
            "T[3].lower".to_string()
        ];
        let output = super::featurizer(sentences, features);
        let expected: Vec<Vec<Vec<String>>> = vec![
            vec![
                vec!["messi".to_string(), "x".to_string()],
                vec!["giành".to_string(), "x".to_string()],
                vec!["quả".to_string(), "x".to_string()]
            ],
            vec![
                vec!["bóng".to_string(), "x".to_string()],
                vec!["vàng".to_string(), "x".to_string()],
            ],
        ];
        assert_eq!(output, expected);
    }
}