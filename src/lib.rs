#[macro_use] extern crate log;
extern crate pretty_env_logger;

mod population;
mod environment;
mod network;
mod genome;
mod layer;
mod cosyne;

pub use genome::Genome;
pub use crate::cosyne::{Cosyne, Config};
pub use environment::Environment;
pub use network::ANN;

pub(crate) use population::Population;