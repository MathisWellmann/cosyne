use crate::network::ANN;

pub trait Environment {
    fn evaluate(&mut self, nn: &mut ANN) -> f64;

    fn reset(&mut self);
}