extern crate rand;
extern crate rand_distr;

use self::rand::{Rng, thread_rng};
use self::rand_distr::{Normal, Uniform};
use crate::genome::{Genome};
use crate::environment::Environment;
use crate::network::ANN;
use crate::Config;

pub struct Population {
    config: Config,
    pub(crate) genomes: Vec<Genome>,
    mutation_prob: f64,
    perturb_prob: f64,
    mutation_strength: f64,
    offspring: Vec<Vec<f64>>,
}

impl Population {
    pub fn new(config: Config, nn: &ANN) -> Population {
        let mut population = Vec::new();

        for _i in 0..config.pop_size {
            population.push(Genome::new(nn.randomize()));
        }

        return Population{
            config,
            genomes: population,
            mutation_prob: 0.3,  // changes over time
            mutation_strength: 1.0,
            perturb_prob: 0.0,
            offspring: Vec::new(),
        }
    }

    // propagate the new input data through the population and evolve
    pub fn generation(&mut self, env: &Box<dyn Environment>, gen: usize, max_gen: usize) -> Genome {
        self.adjust_probs(gen, max_gen);
        self.spawn_offspring();

        let mut best_genome = self.genomes[0].clone();
        for i in 0..self.genomes.len() {
            let fit = env.evaluate(&mut self.genomes[i].network);
            self.genomes[i].update_fitness(fit);

            if fit > best_genome.fitness {
                best_genome = self.genomes[i].clone()
            }

            self.genomes[i].replace_and_permute(&self.offspring[i]);
        }

        return best_genome
    }

    // offspring creates new individuals from top 25% of population
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

    // creates a baby from the genes of two genomes by randomly choosing either from one or the other parent
    fn crossover(&mut self, p1_index: usize, p2_index: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let baby: Vec<f64> = self.genomes[p1_index].genes.iter()
            .zip(&self.genomes[p2_index].genes)
            .map(|(g1, g2)| {
                if rng.gen::<bool>() {
                    *g1
                } else {
                    *g2
                }
            })
            .collect();

        baby
    }

    // assign a random value to genes with probability (inplace)
    fn mutate(&self, genes: &mut Vec<f64>) {
        let d = Normal::new(0.0, 0.4).unwrap();
        let mut rng = rand::thread_rng();
        genes.iter_mut().for_each(|g| {
            if rng.gen::<f64>() < self.mutation_prob {
                if rng.gen::<f64>() < self.perturb_prob {
                    *g += rng.sample(d) * self.mutation_strength;
                } else {
                    *g = rng.gen::<f64>() * self.mutation_strength;
                }
            }
        });
    }

    /// adjust the various probabilities based on current generation and max generation
    fn adjust_probs(&mut self, gen: usize, max_gen: usize) {
        let x: f64 = gen as f64 / max_gen as f64;
        self.mutation_prob = 0.3 - 0.2 * x;
        self.perturb_prob = x;
        self.mutation_strength = 1.0 - 0.9 * x;
    }
}

pub fn sort(vals: Vec<(f64, usize)>) -> Vec<(f64, usize)> {
    if vals.len() <= 1 {
        return vals
    }

    let median = vals[vals.len() / 2].0;

    let mut low_part: Vec<(f64, usize)> = Vec::new();
    let mut high_part: Vec<(f64, usize)> = Vec::new();
    let mut middle_part: Vec<(f64, usize)> = Vec::new();

    for i in vals {
        if i.0 < median {
            low_part.push(i);
        } else if i.0 > median {
            high_part.push(i);
        } else {
            middle_part.push(i);
        }
    }

    low_part = sort(low_part);
    high_part = sort(high_part);

    for i in 0..middle_part.len() {
        low_part.push(middle_part[i]);
    }
    for i in 0..high_part.len() {
        low_part.push(high_part[i]);
    }

    return low_part
}
