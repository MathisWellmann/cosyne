extern crate rand;

use self::rand::{thread_rng, Rng};
use crate::network::ANN;


#[derive(Debug, Clone)]
pub struct Genome {
    pub genes: Vec<f64>,
    pub network: ANN,
    pub fitness: f64,  // this is the fitness of entire genome
    pub fit_enc: Vec<f64>,  // encoded fitness for each weight and bias of network
    pub fit_total: f64,  // sum of fit_enc
    pub fit_min: f64,  // min of fit_enc
    pub fit_max: f64,  // max of fit_enc
}

impl Genome {
    pub fn new(network: ANN) -> Self {
        let genes = network.genes();
        let len_genes = genes.len();
        Genome{
            genes,
            network,
            fitness: 0.0,
            fit_enc: vec![0.0; len_genes],
            fit_total: 0.0,
            fit_min: 0.0,
            fit_max: 0.0,
        }
    }

    pub fn update_fitness(&mut self, fit: f64) {
        let enc_fit = self.network.enc_fitness(fit);
        for i in 0..self.fit_enc.len() {
            self.fit_enc[i] = enc_fit[i];
            self.fit_total += enc_fit[i];
            if enc_fit[i] > self.fit_max {
                self.fit_max = enc_fit[i];
            }
            if enc_fit[i] < self.fit_min {
                self.fit_min = enc_fit[i];
            }
        }
        self.fitness = fit;
    }

    // observe environment and create the signal by forwarding through network
    pub fn create_signal(&mut self, obs: Vec<f64>) -> f64 {
        return self.network.forward(obs)[0];
    }

    pub fn replace_and_permute(&mut self, o: &Vec<f64>) {
        let mut fs = self.fit_enc.clone();
        // lower values will have lower indices
        fs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // TODO: set threshold with config

        let threshold: f64 = fs[(fs.len() as f64 * 0.75).round() as usize];

        // assign permutation probability to each bit
        let mut marked_indices: Vec<usize> = Vec::new();
        let mut rng = thread_rng();

        for i in 0..self.fit_enc.len() {
            if self.fit_enc[i] < threshold {
                // replace with offspring
                self.genes[i] = o[i];
                continue
            }
            let p = 1.0 - ((self.fit_enc[i] - self.fit_min) / self.fit_max - self.fit_min).powf(1.0 / self.genes.len() as f64);
            if rng.gen::<f64>() < p {
                marked_indices.push(i);
            }
        }

        // permute marked by shifting among marked
        let mut temp: f64 = 0.0;
        let mut init: bool = true;
        let mut init_index: usize = 0;
        for i in 0..marked_indices.len() {
            if init {
                init = false;
                temp = self.genes[marked_indices[i] as usize];
                init_index = marked_indices[i] as usize;
                continue
            }

            // swap bits
            self.genes[marked_indices[i] as usize] = temp;
            temp = self.genes[marked_indices[i] as usize];
        }
        // swap first gene with last gene both marked for permutation
        self.genes[init_index] = temp;

        // generate the new network from genes
        self.network.set_genes(&self.genes);
    }

    pub fn kill_and_replace(&mut self, o: &Vec<f64>) {
        self.genes = o.to_vec();
        self.fitness = 0.0;
        self.fit_enc = vec![0.0; self.genes.len()];
        self.fit_total = 0.0;
        self.fit_min = 0.0;
        self.fit_max = 0.0;
    }

}
