use crate::{Config, Environment, Population, ANN};

#[cfg(feature = "plot")]
use {crate::plot_values, failure::Error};

/// The main optimization struct
pub struct Cosyne {
    env: Box<dyn Environment>,
    pop: Population,
    generation: usize,
    champion_fit_history: Vec<f64>,
}

impl Cosyne {
    /// Set the mutation probability of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_prob(&mut self, mp: f64) {
        debug_assert!(mp >= 0.0 && mp <= 1.0);
        self.pop.set_mutation_prob(mp);
    }

    /// Set the mutation strength of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_strength(&mut self, ms: f64) {
        debug_assert!(ms >= 0.0 && ms <= 1.0);
        self.pop.set_mutation_strength(ms);
    }

    /// Set the perturb probability of the population
    /// panics in debug mode if pp < 0.0 || pp > 1.0
    pub fn set_perturb_prob(&mut self, pp: f64) {
        debug_assert!(pp >= 0.0 && pp <= 1.0);
        self.pop.set_perturb_prob(pp);
    }

    /// Create a new CoSyNE optimizer with a given environment, neural network and config
    pub fn new(env: Box<dyn Environment>, nn: ANN, config: Config) -> Self {
        let pop = Population::new(config.clone(), &nn);
        Self {
            env,
            pop,
            generation: 0,
            champion_fit_history: vec![],
        }
    }

    /// Perform an evolutionary step
    pub fn evolve(&mut self) {
        self.pop.evolve(&self.env);
        self.champion_fit_history.push(self.champion().1);

        info!(
            "gen {}, champion fitness: {:.4}",
            self.generation,
            self.champion().1
        );
        self.generation += 1;
    }

    /// Get the current champion and its fitness
    pub fn champion(&self) -> &(ANN, f64) {
        &self.pop.champion()
    }

    #[cfg(feature = "plot")]
    /// Plots the historical fitness values of the population
    pub fn plot_fitness_history(
        &self,
        filename: &str,
        resolution: (u32, u32),
    ) -> Result<(), Error> {
        // TODO: plot worst and average fitness as well

        plot_values(&self.champion_fit_history, filename, resolution)
    }
}
