extern crate nalgebra as na;

#[cfg(test)]
mod tests {
    use super::*;
    use na::Vector2;
    use na::SMatrix;
    #[test]
    fn test_crf() {
        
        // example for multiple two matrix using rust nalgebra
        let a = SMatrix::<f64, 2, 2>::new(1.0, 2.0, 3.0, 4.0);
        let b = SMatrix::<f64, 2, 2>::new(1.0, 2.0, 3.0, 4.0);
        let c = a * b;
        assert_eq!(c, SMatrix::<f64, 2, 2>::new(7.0, 10.0, 15.0, 22.0));
    }
}
