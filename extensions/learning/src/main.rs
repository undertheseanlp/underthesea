// Define struct to represent the CRF model
// struct CRF {
//     // Model parameters
//     weights: Vec<f64>,
//     feature_templates: Vec<FeatureTemplate>,
// }

// // Define struct to represent a feature template
// struct FeatureTemplate {
//     // Feature template parameters
// }

// struct Label {}

// struct Observation {}

// impl CRF {
//     fn calculate_prob(obs: &[Observation], label: &[Label], weights: &[f64]) -> f64 {
//         let mut prob = 0.0;
//         for i in 0..obs.len() {
//             for j in 0..label.len() {
//                 for k in 0..feature_templates.len() {
//                     let feature_value = calculate_feature_value(obs, label, i, j, k);
//                     prob += weights[k] * feature_value;
//                 }
//             }
//         }
//         prob
//     }
// }

// // Define a method for training the CRF model
// impl CRF {
//     fn optimize(
//         weights: &mut [f64],
//         log_likelihood: &dyn Fn(&[f64]) -> f64,
//         gradients: &dyn Fn(&[f64]) -> Vec<f64>,
//     ) {
//         let learning_rate = 0.1;
//         let max_iterations = 1000;
//         for i in 0..max_iterations {
//             let grad = gradients(weights);
//             for j in 0..weights.len() {
//                 weights[j] -= learning_rate * grad[j];
//             }
//             let current_log_likelihood = log_likelihood(weights);
//             // check convergence
//         }
//     }

//     fn train(&mut self, observations: &[Vec<Observation>], labels: &[Vec<Label>]) {
//         // Initialize the model parameters
//         self.weights = vec![0.0; self.feature_templates.len()];

//         // Define a function to calculate the log-likelihood of the observed data given the model parameters
//         let log_likelihood = |weights: &[f64]| {
//             // Calculate the log-likelihood
//             let mut log_likelihood = 0.0;
//             for (obs, label) in observations.iter().zip(labels.iter()) {
//                 log_likelihood += self.calculate_prob(obs, label, weights);
//             }
//             log_likelihood
//         };

//         // Define a function to calculate the gradients of the log-likelihood with respect to the model parameters
//         let gradients = |weights: &[f64]| -> Vec<f64> {
//             // Calculate the gradients
//             vec![]
//         };

//         // Use an optimization algorithm, such as gradient descent, to find the optimal model parameters
//         optimize(&mut self.weights, &log_likelihood, &gradients);
//     }
// }

// // Define a method for making predictions with the trained CRF model
// impl CRF {
//     fn predict(&self, observations: &[Observation]) -> Vec<Label> {
//         // Use the Viterbi algorithm to find the most likely label sequence given the observations and the model parameters
//         let mut label_sequence = vec![];
//         return label_sequence;
//     }
// }
use csv::Reader;
use std::fs::File;
use learning::crf::crf::Token;

fn read_training_data(file_path: &str) -> (Vec<String>, Vec<String>) {
    let file  = File::open(file_path).expect("Failed to open file");
    let mut rdr = Reader::from_reader(file);
    let mut observations = vec![];
    let mut labels = vec![];
    for result in rdr.records() {
        let record = result.expect("Failed to parse record");
        let label_string = &record[0];
        let label = label_string.chars().skip(9).collect::<String>();
        let input = record[1].to_string();
        observations.push(input);
        labels.push(label);
    }
    (observations, labels)
}

fn main() {
    println!("Hello, world!");
    let training_data = read_training_data("/Users/anhv/projects/undertheseanlp/underthesea/extensions/learning/src/train.csv");
    let (observations, labels) = training_data;

    for (i, observation) in observations.iter().enumerate() {
        println!("Observation {}: {}", i, observation);
        println!("Label: {}", labels[i]);
    }
}
