#[derive(Debug, Clone)]
/// Configuration of CoSyNE
pub struct Config {
    pub pop_size: usize,         // total population size
    pub num_species: usize,      // number of sub-populations
    pub elite_threshold: f64,    // Threshold defining
    pub mutation_prob: f64,
    pub mutation_strength: f64,
    pub perturb_prob: f64,
}

impl Config {
    /// Create a new Config
    pub fn new(pop_size: usize, num_species: usize) -> Self {
        Self {
            pop_size,
            num_species,
            elite_threshold: 0.25,
            mutation_prob: 0.3,
            mutation_strength: 0.5,
            perturb_prob: 0.5,
        }
    }

    /// Set the elite_threshold
    /// which defines what percentage of sorted population will be regarded for offspring creation
    /// panics if probability is < 0.0 || t > 1.0
    pub fn set_elite_threshold(&mut self, t: f64) {
        assert!(t >= 0.0 && t <= 1.0);
        self.elite_threshold = t;
    }
}
