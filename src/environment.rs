use crate::network::ANN;

pub trait Environment {
    fn evaluate(&self, nn: &mut ANN) -> f64;
}