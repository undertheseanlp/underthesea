//! Cosine similarity
//!
//! Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.
//!
//! # Author: Vu Anh
//! # Date: 2023-07-30
use nalgebra::DVector;

pub fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let va = DVector::from_vec(a.clone());
    let vb = DVector::from_vec(b.clone());

    va.dot(&vb) / (va.norm() * vb.norm())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_1() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        assert!((cosine_similarity(&a, &b)).abs() < f64::EPSILON);
    }
}