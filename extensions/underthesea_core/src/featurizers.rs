use regex::{Regex};
use std::collections::HashSet;


/* Struct for FeatureTemplate
///
/// Token syntax
/// ===========================
///          _ offset 1
///         /  _ offset 2
///        /  /  _ column
///       /  /  /
///     T[0,2][0].is_digit
///               /__ function
///
/// ===========================
/// Sample tagged sentence
/// ===========================
/// this     A
/// is       B
/// a        C
/// sample   D
/// sentence E
/// ===========================
/// offset1, offset2 and column may contains negative value (for offset value)
/// Supported functions: lower, isdigit, istitle, is_in_dict
*/
#[derive(Debug)]
pub struct FeatureTemplate {
    syntax: String,
    offset1: isize,
    offset2: Option<isize>,
    column: isize,
    function: Option<String>,
}

pub struct CRFFeaturizer {
    pub feature_configs: Vec<String>,
    pub dictionary: HashSet<String>,
    feature_templates: Vec<FeatureTemplate>,
}


impl CRFFeaturizer {
    pub fn new(
        feature_configs: Vec<String>,
        dictionary: HashSet<String>,
    ) -> Self {
        let mut feature_templates: Vec<FeatureTemplate> = Vec::new();
        let re = Regex::new(
            r"T\[(?P<index1>-?\d+)(,(?P<index2>-?\d+))?](\[(?P<column>.*)])?(\.(?P<function>.*))?"
        ).unwrap();
        for feature_config in &feature_configs {
            let mut feature_template = FeatureTemplate {
                syntax: String::from(""),
                offset1: 0,
                offset2: None,
                column: 0,
                function: None,
            };
            feature_template.syntax = String::from(feature_config);
            for cap in re.captures_iter(feature_config.as_str()) {
                match cap.name("index1") {
                    Some(s) => {
                        feature_template.offset1 = s.as_str().parse::<isize>().unwrap();
                    }
                    _ => ()
                }
                match cap.name("index2") {
                    Some(s) => {
                        feature_template.offset2 = Option::from(s.as_str().parse::<isize>().unwrap());
                    }
                    _ => ()
                }
                match cap.name("function") {
                    Some(s) => {
                        feature_template.function = Option::from(String::from(s.as_str()));
                    }
                    _ => ()
                }
            }
            feature_templates.push(feature_template);
        }
        CRFFeaturizer {
            feature_configs,
            dictionary,
            feature_templates,
        }
    }

    /// generate features for each token <position>-th in sentence
    /// Sentence Example
    /// Messi   X
    /// giành   X
    /// quả     X
    pub fn generate_token_features(&self, sentence: &Vec<Vec<String>>, position: usize) -> Vec<String> {
        let mut features = Vec::new();
        for feature_template in &self.feature_templates {
            let index1 = position as isize + feature_template.offset1;
            let bos_value = String::from(&feature_template.syntax) + "=" + "BOS";
            let eos_value = String::from(&feature_template.syntax) + "=" + "EOS";
            let column = feature_template.column;
            let n = sentence.len() as isize;
            let mut text: String;
            if index1 < 0 {
                features.push(bos_value);
                continue;
            } else if index1 >= n {
                features.push(eos_value);
                continue;
            } else {
                text = String::from(&sentence[index1 as usize][column as usize]);
            }

            match feature_template.offset2 {
                None => {}
                Some(offset2) => {
                    let index2 = position as isize + offset2;
                    if index2 < 0 {
                        features.push(bos_value);
                        continue;
                    } else if index2 >= n {
                        features.push(eos_value);
                        continue;
                    } else {
                        for i in index1 + 1..index2 + 1 {
                            text = text + " " + &sentence[i as usize][column as usize];
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
                        "isdigit" => {
                            let is_digit = text.parse::<i32>();
                            match is_digit {
                                Ok(_) => { text = String::from("True") }
                                Err(_) => { text = String::from("False") }
                            }
                        }
                        "istitle" => {
                            let mut is_title = "True";
                            for part in text.split(" ") {
                                let first_char = String::from(part.chars().nth(0).unwrap());
                                if first_char != first_char.to_uppercase() {
                                    is_title = "False";
                                    break;
                                }
                            }
                            text = String::from(is_title);
                        }
                        "is_in_dict" => {
                            if self.dictionary.contains(text.to_lowercase().as_str()) {
                                text = String::from("True");
                            } else {
                                text = String::from("False");
                            }
                        }
                        _ => {}
                    }
                }
            }
            let value = String::from(&feature_template.syntax) + "=" + text.as_str();
            features.push(value);
        }
        return features;
    }
    pub fn process(&self, sentences: Vec<Vec<Vec<String>>>) -> Vec<Vec<Vec<String>>> {
        let mut sentences_features = Vec::new();
        for sentence in sentences {
            // generate features for each sentence
            let mut sentence_features = Vec::new();
            for position in 0..sentence.len() {
                let token_features = self.generate_token_features(&sentence, position);
                sentence_features.push(token_features);
            }
            sentences_features.push(sentence_features);
        }
        return sentences_features;
    }
}

