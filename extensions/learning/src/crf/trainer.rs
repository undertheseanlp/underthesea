// create struct Trainer
struct Trainer {   
}

// implement method train for Trainer
impl Trainer {
    fn train(&mut self, crf: &mut CRF, tokens: Vec<Token>, labels: Vec<String>) {
        // do something
    }
}
extern crate nalgebra as na;
use na::Vector2;

// create a vector using Vector2
let v = Vector2::new(1.0, 2.0);
