#![deny(missing_docs, rustdoc::missing_crate_level_docs)]

//! CoSyNE crate for optimizing fixed topology neural network using co-evolution
//! as described in the original paper (see resources/CoSyNE.pdf)

#[macro_use]
extern crate log;
extern crate pretty_env_logger;

extern crate nalgebra as na;

mod activation;
mod config;
mod cosyne;
mod layer;
mod network;
mod permutation_prob_f;
#[cfg(feature = "plot")]
mod plot;
mod population;

pub use crate::cosyne::Cosyne;
pub use activation::Activation;
pub use config::Config;
pub use network::ANN;
pub use permutation_prob_f::PermutationProbF;
pub use population::Population;

pub(crate) use layer::Layer;

#[cfg(feature = "plot")]
pub(crate) use plot::plot_values;

/// Environment to test the neural network in
pub trait Environment {
    /// Return the fitness of a given neural network in the environment.
    /// Higher values indicate a more fit candidate
    fn evaluate(&self, nn: &mut ANN) -> f64;
}
