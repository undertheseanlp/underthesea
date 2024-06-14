extern crate ndarray;

use ndarray::{Array1, Array2};
use ndarray::prelude::*;

pub struct LogisticRegression {
    weights: Array1<f64>,
    learning_rate: f64,
    iterations: usize,
}

impl LogisticRegression {
    // Constructor
    pub fn new() -> Self {
        LogisticRegression {
            weights: Array1::zeros(1),
            learning_rate: 0.01,
            iterations: 1000,
        }
    }

    // Method to set learning_rate and iterations
    pub fn with_hyperparams(mut self, learning_rate: f64, iterations: usize) -> Self {
        self.learning_rate = learning_rate;
        self.iterations = iterations;
        self
    }

    // Sigmoid function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    pub fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) {
        let m = x_train.nrows(); // number of samples
        let n = x_train.ncols(); // number of features
        self.weights = Array1::zeros(n); // initializing weights
    
        // Gradient Descent
        for _ in 0..self.iterations {
            let mut gradient = Array1::zeros(n); // initialize gradient
    
            // calculate gradient for each sample
            for j in 0..m {
                let x = x_train.row(j).to_owned();
                let h = Self::sigmoid(x.dot(&self.weights));
                let error = h - y_train[j];
                gradient = gradient + error * &x;
            }
    
            // update weights
            self.weights = &self.weights - self.learning_rate * gradient / m as f64;
        }
    }

    // Predict function for predicting an output with the learned weights
    pub fn predict(&self, x: &Array1<f64>) -> f64 {
        let z = x.dot(&self.weights);
        Self::sigmoid(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn logistic_regression_test() {
        // Initialize logistic regression model
        let mut model = LogisticRegression::new().with_hyperparams(0.01, 20000);

        // Training data
        let x_train = array![[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]];
        let y_train = array![0., 0., 1., 1.];

        // Fit model
        model.fit(&x_train, &y_train);

        // Test data
        let x_test = array![2., 3., 4.];

        // Predict
        let prediction = model.predict(&x_test);

        // Test that the model's prediction is close to the expected value
        print!("Error {}", (prediction - 1.).abs());
        assert!((prediction - 0.1).abs() < 1.0);
    }
}