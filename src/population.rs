extern crate rand;
extern crate rand_distr;

use self::rand::Rng;
use self::rand_distr::{Normal, Distribution};
use crate::genome::{Genome};
use crate::environment::Environment;
use crate::network::ANN;

pub struct Population {
    generation_counter: i64,
    pub(crate) genomes: Vec<Genome>,
    mutation_prob: f64,
    offspring: Vec<Vec<f64>>,
}

impl Population {
    pub fn new(pop_size: usize, nn: &ANN) -> Population {
        let mut population = Vec::new();

        for _i in 0..pop_size {
            population.push(Genome::new(nn.randomize()));
        }

        return Population{
            generation_counter: 0,
            genomes: population,
            mutation_prob: 1.0,  // changes over time
            offspring: Vec::new(),
        }
    }

    // propagate the new input data through the population and evolve
    pub fn generation(&mut self, env: &Box<dyn Environment>) -> Genome {
        self.adjust_mutation_rate();
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

        self.generation_counter += 1;

        return best_genome
    }

    // offspring creates new individuals from top 25% of population
    fn spawn_offspring(&mut self) {
        let mut fs: Vec<f64> = Vec::new();
        for i in 0..self.genomes.len() {
            fs.push(self.genomes[i].fitness);
        }
        let sorted = sort_vec(fs);
        let fit_threshold = sorted[(sorted.len() as f64 * 0.75).round() as usize];

        let mut best_indices: Vec<usize> = Vec::new();
        for i in 0..self.genomes.len() {
            if self.genomes[i].fitness >= fit_threshold {
                // add index of individual
                best_indices.push(i);
            }
        }

        self.offspring.clear();

        let mut rng = rand::thread_rng();
        // fill offspring population with new children
        while self.offspring.len() < self.genomes.len() {
            // take two random indices in top 25% of population

            let parent1 = best_indices[rng.gen_range(0, best_indices.len())];
            let parent2 = best_indices[rng.gen_range(0, best_indices.len())];
            let (child1, child2) = self.crossover(parent1, parent2);

            let m_child1 = self.mutate(child1);
            let m_child2 = self.mutate(child2);

            self.offspring.push(m_child1);
            self.offspring.push(m_child2);
        }
    }

    // creates two children using the indices of two parents using single point crossover
    fn crossover(&mut self, p1_index: usize, p2_index: usize) -> (Vec<f64>, Vec<f64>) {
        let mut child1: Vec<f64> = Vec::new();
        let mut child2: Vec<f64> = Vec::new();

        let crossover_point = self.genomes[p1_index].genes.len() / 2;
        for i in 0..self.genomes[p1_index].genes.len() {
            if i < crossover_point {
                child1.push(self.genomes[p1_index].genes[i]);
                child2.push(self.genomes[p2_index].genes[i]);
                continue
            }
            child1.push(self.genomes[p2_index].genes[i]);
            child2.push(self.genomes[p1_index].genes[i]);
        }
        return (child1, child2)
    }

    // assign a random value to child with probability (inplace)
    fn mutate(&self, mut child: Vec<f64>) -> Vec<f64> {
        let n = Normal::new(0.0, 0.4).unwrap();
        let mut rng = rand::thread_rng();
        for i in 0..child.len() {
            if n.sample(&mut rng) < self.mutation_prob {
                child[i] = n.sample(&mut rng);
            }
        }
        return child
    }

    fn adjust_mutation_rate(&mut self) {
        self.mutation_prob = (-0.001 * self.generation_counter as f64).exp();
        if self.mutation_prob < 0.01 {
            self.mutation_prob = 0.01;
        }
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

pub fn sort_vec(vals: Vec<f64>) -> Vec<f64> {
    if vals.len() <= 1 {
        return vals
    }

    let median = vals[vals.len() / 2];

    let mut low_part: Vec<f64> = Vec::new();
    let mut high_part: Vec<f64> = Vec::new();
    let mut middle_part: Vec<f64> = Vec::new();

    for i in vals {
        if i < median {
            low_part.push(i);
        } else if i > median {
            high_part.push(i);
        } else {
            middle_part.push(i);
        }
    }

    low_part = sort_vec(low_part);
    high_part = sort_vec(high_part);

    for i in 0..middle_part.len() {
        low_part.push(middle_part[i]);
    }
    for i in 0..high_part.len() {
        low_part.push(high_part[i]);
    }

    return low_part
}
