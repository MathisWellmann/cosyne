use crate::permutation_prob_f::PermutationProbF;

#[derive(Debug, Clone, Copy)]
/// Configuration of CoSyNE
pub struct Config {
    /// total number of sub-populations
    pub pop_size: usize,
    /// Threshold defining how much of the best performing
    pub top_ratio_to_recombine: f64,
    /// probability of mutation a weight of a sub-population
    pub mutation_prob: f64,
    /// strength of the mutation
    pub mutation_strength: f64,
    /// probability of applying a perturbation
    pub perturb_prob: f64,
    /// Permutation function to use
    pub permutation_prob_f: PermutationProbF,
}

impl Config {
    /// Create a new Config
    pub fn new(pop_size: usize) -> Self {
        Self {
            pop_size,
            top_ratio_to_recombine: 0.25,
            mutation_prob: 0.3,
            mutation_strength: 0.5,
            perturb_prob: 0.5,
            permutation_prob_f: PermutationProbF::Uniform(1.0),
        }
    }
}
