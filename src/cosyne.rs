use crate::{Genome, Activation, Environment, ANN, Population};

pub struct Cosyne{
    env: Box<dyn Environment>,
    start_nn: ANN,
    config: Config,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) pop_size: usize,
    activation: Option<Activation>,
    activations: Vec<Activation>,
    pub(crate) elite_threshold: f64,
}

impl Config {
    /// Creates a new Config with a fixed activation
    pub fn new_fixed_activation(pop_size: usize, activation: Activation) -> Self {
        Self {
            pop_size,
            activation: Some(activation),
            activations: vec![],
            elite_threshold: 0.5,
        }
    }

    /// creates a new Config with a vector of activations which will be used in the evolution process
    pub fn new_with_activations(pop_size: usize, activations: Vec<Activation>) -> Self {
        // TODO: remove once feature ready
        unimplemented!("random activations have not been integrated yet!");
        Self {
            pop_size,
            activation: None,
            activations,
            elite_threshold: 0.5,
        }
    }

    /// sets the elite_threshold
    pub fn set_elite_threshold(&mut self, t: f64) {
        assert!(t >= 0.0);
        assert!(t < 1.0);
        self.elite_threshold = t;
    }
}

impl Cosyne {
    pub fn new(env: Box<dyn Environment>, nn: ANN, config: Config) -> Self {
        Self {
            env,
            start_nn: nn,
            config,
        }
    }

    /// optimize the neural network for n generations.
    /// Returns the champion Genome
    pub fn optimize(&self, generations: usize) -> Genome {
        let mut pop = Population::new(self.config.clone(), &self.start_nn);
        let mut champion: Genome = pop.genomes[0].clone();
        for g in 0..generations {
            let best = pop.generation(&self.env, g, generations);
            if best.fitness > champion.fitness {
                champion = best;
            }
            debug!("gen {}, champion fitness: {:.4}", g, champion.fitness);
        }

        champion
    }
}