use crate::network::ANN;

pub trait Environment {
    fn evaluate(&mut self, nn: &ANN) -> f64;
}