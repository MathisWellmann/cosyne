#[derive(Debug, Clone, Copy)]
/// Enumerate the probability functions used for marking weights for permutation
pub enum PermutationProbF {
    /// Each weight has the same probability of being marked for permutation
    Uniform(f64),
    /// Each weight has a probability of being permuted relative to it's peers in sub-population
    /// not quite the same as in paper
    Relative,
}

impl PermutationProbF {
    /// Return the probability that a weight is marked for permutation
    /// given the sub-population fitnesses
    pub(crate) fn get_probability(&self, subpopulation_fits: &Vec<f64>, weight_fit: f64) -> f64 {
        match self {
            PermutationProbF::Uniform(p) => *p,
            PermutationProbF::Relative => {
                let mut min_fit: f64 = subpopulation_fits[0];
                let mut max_fit: f64 = subpopulation_fits[0];
                for f in subpopulation_fits {
                    if *f < min_fit {
                        min_fit = *f;
                    } else if *f > max_fit {
                        max_fit = *f;
                    }
                }

                1.0 - scale(min_fit, max_fit, 0.0, 1.0, weight_fit)
            }
        }
    }
}

/// Scale a value from one range to another
pub fn scale(from_min: f64, from_max: f64, to_min: f64, to_max: f64, value: f64) -> f64 {
    to_min + ((value - from_min) * (to_max - to_min)) / (from_max - from_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use round::round;

    #[test]
    fn permutation_prob_f() {
        let ppf = PermutationProbF::Uniform(1.0);
        assert_eq!(ppf.get_probability(&vec![], 0.5), 1.0);

        let ppf = PermutationProbF::Relative;
        let spf: Vec<f64> = vec![0.1, 0.2, 0.5, -0.1, -0.2];
        let weight_fit: f64 = spf[0];
        let prob: f64 = ppf.get_probability(&spf, weight_fit);
        assert_eq!(round(prob, 3), 0.571);
    }
}
