#[macro_use]
extern crate log;
extern crate pretty_env_logger;

mod activation;
mod config;
mod cosyne;
mod environment;
mod genome;
mod layer;
mod network;
mod population;
#[cfg(feature = "plot")]
mod plot;

pub use activation::Activation;
pub use config::Config;
pub use crate::cosyne::Cosyne;
pub use environment::Environment;
pub use genome::Genome;
pub use network::ANN;

pub(crate) use population::Population;
pub(crate) use layer::Layer;

#[cfg(feature = "plot")]
pub(crate) use plot::plot_multiple_series;
