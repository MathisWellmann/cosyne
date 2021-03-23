use crate::{Config, Environment, ANN};
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Uniform};

pub(crate) const DEFAULT_FIT: f64 = std::f64::MIN;

pub struct Population {
    config: Config,
    network_topology: ANN,
    // stores an n x m matrix of chromosome value and corresponding fitness
    // NOTE: indexing is reverse of what the paper refers to:
    // sub_populations[j] is a complete network
    // sub_populations[j][i] is the ith weight of the jth network
    sub_populations: Vec<Vec<(f64, f64)>>,
    // number of species (components, sub-populations) or just weights and biases of the network topology
    n: usize,
    // m sub-genotypes or population size,
    // so for each weight (or bias) in the nn topology there are m different weight (or bias) variations
    m: usize,
    // Network with fitness
    champion: (ANN, f64),
    current_generation: usize,
}

impl Population {
    /// Set the mutation probability of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_prob(&mut self, mp: f64) {
        self.config.mutation_prob = mp;
    }

    /// Set the mutation strength of the population
    /// panics in debug mode if mp < 0.0 || mp > 1.0
    pub fn set_mutation_strength(&mut self, ms: f64) {
        self.config.mutation_strength = ms;
    }

    /// Set the perturb probability of the population
    /// panics in debug mode if pp < 0.0 || pp > 1.0
    pub fn set_perturb_prob(&mut self, pp: f64) {
        self.config.perturb_prob = pp;
    }

    /// Return a reference to current champion in population
    pub(crate) fn champion(&self) -> &(ANN, f64) {
        &self.champion
    }

    /// Create a new population with a given config and network
    pub(crate) fn new(config: Config, nn: &ANN) -> Population {
        let n: usize = nn.num_genes();
        let m: usize = config.pop_size;

        let mut sub_populations: Vec<Vec<(f64, f64)>> = Vec::new();
        for _i in 0..config.pop_size {
            let genes: Vec<f64> = nn.randomize().genes();
            let mut chromosome: Vec<(f64, f64)> = Vec::with_capacity(n);
            for v in &genes {
                chromosome.push((*v, DEFAULT_FIT));
            }
            sub_populations.push(chromosome);
        }

        return Population {
            config,
            network_topology: nn.clone(),
            sub_populations,
            n,
            m,
            champion: (nn.clone(), std::f64::MIN),
            current_generation: 0,
        };
    }

    /// Perform a single generational evolutionary step in a given environment
    pub(crate) fn evolve(&mut self, env: &Box<dyn Environment>) {
        // evaluate entire population
        let mut fits: Vec<f64> = Vec::with_capacity(self.m);
        for j in 0..self.m {
            // get the genes of network chromosome
            let chromosome: &Vec<(f64, f64)> = &self.sub_populations[j];
            let genes: Vec<f64> = chromosome.iter().map(|(v, _)| *v).collect();
            // transform chromosome onto network topology
            self.network_topology.set_genes(&genes);
            // evaluate the fitness in environment
            let fit = env.evaluate(&mut self.network_topology);
            fits.push(fit);
            if fit > self.champion.1 {
                // save the champion with fitness
                self.champion = (self.network_topology.clone(), fit);
            }
        }
        self.update_fitnesses(&fits);

        let offspring = self.spawn_offspring();

        self.replace_and_permute(&offspring);
    }

    /// Update existing chromosome fitnesses with the new network fits
    fn update_fitnesses(&mut self, new_fits: &Vec<f64>) {
        let g: f64 = self.current_generation as f64;
        for (j, new_fit) in new_fits.iter().enumerate() {
            self.sub_populations[j].iter_mut().for_each(|(_, old_fit)| {
                *old_fit *= g; // undo the mean
                *old_fit += new_fit; // add new fitness
                *old_fit /= g + 1.0; // redo the mean
            });
        }
        self.current_generation += 1;
    }

    /// Create offspring population from top n% of population
    fn spawn_offspring(&mut self) -> Vec<Vec<(f64, f64)>> {
        // find parents with highest mean fitness based on elite_threshold in config
        // compute mean fitness of each column (network mean)
        let mut mean_fits: Vec<(usize, f64)> = Vec::with_capacity(self.m);
        for j in 0..self.m {
            let mean_fit: f64 =
                self.sub_populations[j].iter().map(|(_, f)| *f).sum::<f64>() / self.n as f64;
            mean_fits.push((j, mean_fit));
        }
        // sort mean_fits by fitness, lower indices will have lower fitness
        mean_fits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let elite_threshold: usize =
            (self.m as f64 * self.config.top_ratio_to_recombine).floor() as usize;
        let mut o: Vec<Vec<(f64, f64)>> = Vec::with_capacity(
            (self.config.top_ratio_to_recombine * self.m as f64).ceil() as usize,
        );
        for (j, _f) in mean_fits.iter().take(elite_threshold) {
            o.push(self.sub_populations[*j].clone());
        }

        self.crossover(&mut o);

        self.mutate(&mut o);

        o
    }

    /// Perform crossover mutation operator on offspring population,
    fn crossover(&mut self, o: &mut Vec<Vec<(f64, f64)>>) {
        // create vec of deranged indices, not sure if actually good but should be good enough
        let mut rng = thread_rng();
        let d = Uniform::new(0, o.len());
        let mut deranged: Vec<usize> = vec![];
        for i in 0..o.len() {
            let idx: usize = rng.sample(d);
            if i != idx {
                deranged.push(idx);
            } else {
                deranged.push((idx + 1) % o.len());
            }
        }

        let d = Uniform::new(0, self.n);
        for (p1, p2) in (0..o.len()).zip(&deranged) {
            // TODO: different user defined crossover methods
            let crossover_point: usize = rng.sample(d);

            for i in 0..crossover_point {
                // switch chromosomes and fitness between parent 1 and 2
                let old_p1_vals = self.sub_populations[p1][i];
                self.sub_populations[p1][i] = self.sub_populations[*p2][i];
                self.sub_populations[*p2][i] = old_p1_vals;
            }
        }
    }

    /// Perform a mutation operator on offspring population,
    /// by either perturbing or completely replacing values
    fn mutate(&self, o: &mut Vec<Vec<(f64, f64)>>) {
        // TODO: user defined mutation distribution in case of pertubation
        let d = Normal::new(0.0, 0.4).unwrap();
        let mut rng = rand::thread_rng();

        o.iter_mut().flatten().for_each(|(v, _f)| {
            if rng.gen::<f64>() < self.config.mutation_prob {
                if rng.gen::<f64>() < self.config.perturb_prob {
                    *v += rng.sample(d) * self.config.mutation_strength;
                } else {
                    *v = rng.gen::<f64>() * self.config.mutation_strength;
                }
            }
        });
    }

    /// Replace the least fit chromosome in each sub-population with newly created offspring
    /// Also permute the left over original chromosomes among each other in the sub-population
    fn replace_and_permute(&mut self, o: &Vec<Vec<(f64, f64)>>) {
        let mut rng = thread_rng();
        for i in 0..self.n {
            // sort the sub-population
            let mut genes: Vec<(f64, f64)> = Vec::with_capacity(self.m);
            for j in 0..self.m {
                genes.push(self.sub_populations[j][i]);
            }
            genes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let fit_threshold: f64 =
                genes[(self.config.top_ratio_to_recombine * genes.len() as f64).floor() as usize].1;

            let mut offspring_idx: usize = 0;
            for j in 0..self.m {
                if self.sub_populations[j][i].1 < fit_threshold {
                    // replace least fit
                    self.sub_populations[j][i] = o[offspring_idx][i];
                    offspring_idx += 1;
                }
            }

            // assign permutation probability of left over (original) chromosomes in
            // sub-population
            let sub_pop_fits: Vec<f64> =
                (0..self.m).map(|j| self.sub_populations[j][i].1).collect();
            let mut marked: Vec<usize> = vec![];
            for j in 0..self.m {
                let prob: f64 = self
                    .config
                    .permutation_prob_f
                    .get_probability(&sub_pop_fits, self.sub_populations[j][i].1);
                if rng.gen::<f64>() < prob {
                    // mark for permutation
                    marked.push(j);
                }
            }

            if marked.len() == 0 {
                return;
            }

            // permute marked by shifting among them
            let mut temp = self.sub_populations[marked[0]][i];
            for marked_idx in marked.iter().skip(1) {
                // swap
                let old_val = self.sub_populations[*marked_idx][i];
                self.sub_populations[*marked_idx][i] = temp;
                temp = old_val;
            }
            self.sub_populations[marked[0]][i] = temp;
        }
    }
}
