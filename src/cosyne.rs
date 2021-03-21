use crate::{Environment, Genome, Population, ANN, Config};

#[cfg(feature = "plot")]
use {
    crate::plot_multiple_series,
    failure::Error,
};

pub struct Cosyne {
    env: Box<dyn Environment>,
    species: Vec<Population>,
    champion: Genome,
    generation: usize,
    champion_fit_history: Vec<Vec<f64>>,
}


impl Cosyne {
    /// Create a new CoSyNE optimizer with a given environment, neural network and config
    pub fn new(env: Box<dyn Environment>, nn: ANN, config: Config) -> Self {
        let mut species: Vec<Population> = Vec::with_capacity(config.num_species);
        for _ in 0..config.num_species {
            species.push(Population::new(config.clone(), &nn));
        }
        let champion = Genome::new(nn.randomize());
        Self {
            env,
            species,
            champion,
            generation: 0,
            champion_fit_history: vec![vec![]; config.num_species],
        }
    }

    /// Perform an evolutionary step
    pub fn step(&mut self) {
        for (i, p) in self.species.iter_mut().enumerate() {
            let best = p.generation(&self.env);
            if best.fitness > self.champion.fitness {
                self.champion = best;
            }
            self.champion_fit_history[i].push(self.champion.fitness);

            info!(
                "species: {}, gen {}, champion fitness: {:.4}",
                i, self.generation, self.champion.fitness
            );
        }
        self.generation += 1;
    }

    /// Get the current champion
    pub fn champion(&self) -> &Genome {
        &self.champion
    }

    #[cfg(feature = "plot")]
    /// Plots the historical fitness values of the population
    pub fn plot_fitness_history(&self, filename: &str, resolution: (u32, u32)) -> Result<(), Error> {
        // TODO: plot worst and average fitness as well

        plot_multiple_series(&self.champion_fit_history, filename, resolution)
    }
}
