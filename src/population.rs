extern crate rand;
extern crate rand_distr;

use self::rand::{thread_rng, Rng};
use self::rand_distr::{Normal, Uniform};
use crate::environment::Environment;
use crate::genome::Genome;
use crate::network::ANN;
use crate::Config;

pub struct Population {
    config: Config,
    pub(crate) genomes: Vec<Genome>,
    offspring: Vec<Vec<f64>>,
}

impl Population {
    /// Set the mutation probability of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_prob(&mut self, mp: f64) {
        debug_assert!(mp >= 0.0 && mp <= 1.0);
        self.config.mutation_prob = mp;
    }

    /// Set the mutation strength of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_strength(&mut self, ms: f64) {
        debug_assert!(ms >= 0.0 && ms <= 1.0);
        self.config.mutation_strength = ms;
    }

    /// Set the perturb probability of the population
    /// panics in debug mode if pp < 0.0 || pp > 1.0
    pub fn set_perturb_prob(&mut self, pp: f64) {
        debug_assert!(pp >= 0.0 && pp <= 1.0);
        self.config.perturb_prob = pp;
    }

    /// Create a new population with a given config and network
    pub(crate) fn new(config: Config, nn: &ANN) -> Population {
        let mut population = Vec::new();

        for _i in 0..config.pop_size {
            population.push(Genome::new(nn.randomize()));
        }

        return Population {
            config,
            genomes: population,
            offspring: Vec::new(),
        };
    }

    /// Perform a single generational evolutionary step in a given environment
    pub(crate) fn generation(&mut self, env: &Box<dyn Environment>) -> Genome {
        self.spawn_offspring();

        let mut best_genome = self.genomes[0].clone();
        for (i, g) in self.genomes.iter_mut().enumerate() {
            let fit = env.evaluate(&mut g.network);
            g.update_fitness(fit);

            if fit > best_genome.fitness {
                best_genome = g.clone()
            }

            g.replace_and_permute(&self.offspring[i]);
        }

        return best_genome;
    }

    /// Create new individuals from top n% of population
    fn spawn_offspring(&mut self) {
        let mut fs: Vec<f64> = self.genomes.iter().map(|g| g.fitness).collect();
        // lower values have lower indices
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fit_threshold = fs[(fs.len() as f64 * self.config.elite_threshold).round() as usize];

        let mut best_indices: Vec<usize> = Vec::new();
        for (i, g) in self.genomes.iter().enumerate() {
            if g.fitness >= fit_threshold {
                // add index of individual
                best_indices.push(i);
            }
        }

        self.offspring.clear();

        let mut rng = rand::thread_rng();
        let d = Uniform::new(0, best_indices.len());
        // fill offspring population with new children
        while self.offspring.len() < self.genomes.len() {
            // take two random indices in top 25% of population
            let parent1 = best_indices[rng.sample(d)];
            let parent2 = best_indices[rng.sample(d)];
            let mut baby = self.crossover(parent1, parent2);

            self.mutate(&mut baby);

            self.offspring.push(baby);
        }
    }

    /// Perform crossover mutation operator,
    /// creating a baby from the genes of two genomes
    /// by randomly choosing either from one or the other parent
    fn crossover(&mut self, p1_index: usize, p2_index: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let baby: Vec<f64> = self.genomes[p1_index]
            .genes
            .iter()
            .zip(&self.genomes[p2_index].genes)
            .map(|(g1, g2)| if rng.gen::<bool>() { *g1 } else { *g2 })
            .collect();

        baby
    }

    /// Perform a mutation operator,
    /// by either perturbing or completely replacing values
    fn mutate(&self, genes: &mut Vec<f64>) {
        let d = Normal::new(0.0, 0.4).unwrap();
        let mut rng = rand::thread_rng();
        genes.iter_mut().for_each(|g| {
            if rng.gen::<f64>() < self.config.mutation_prob {
                if rng.gen::<f64>() < self.config.perturb_prob {
                    *g += rng.sample(d) * self.config.mutation_strength;
                } else {
                    *g = rng.gen::<f64>() * self.config.mutation_strength;
                }
            }
        });
    }
}
