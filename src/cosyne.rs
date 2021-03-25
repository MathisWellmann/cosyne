use crate::{Config, Environment, Population, ANN};

#[cfg(feature = "plot")]
use {crate::plot_values, failure::Error};
use crate::population::DEFAULT_FIT;

/// The main optimization struct
pub struct Cosyne {
    config: Config,
    env: Box<dyn Environment>,
    pop: Population,
    generation: usize,
    champion_fit_history: Vec<f64>,
    champion: (ANN, f64),   // network with fitness
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
        let champion = (pop.get_network(0), DEFAULT_FIT);
        Self {
            config,
            env,
            pop,
            generation: 0,
            champion_fit_history: vec![],
            champion,
        }
    }

    /// Perform an evolutionary step
    pub fn evolve(&mut self) {
        // evaluate entire population
        let mut fits: Vec<f64> = Vec::with_capacity(self.config.pop_size);
        for j in 0..self.config.pop_size {
            // get the genes of network chromosome
            let mut net = self.pop.get_network(j);
            // evaluate the fitness in environment
            let fit = self.env.evaluate(&mut net);
            fits.push(fit);
            if fit > self.champion.1 {
                // save the champion with fitness
                self.champion = (net.clone(), fit);
            }
        }
        self.pop.update_fitnesses(&fits);

        self.pop.evolve();
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
        &self.champion
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
