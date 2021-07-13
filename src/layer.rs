use rand::{thread_rng, Rng};

use ndarray::{Array as NewMatrix, Array2 as Matrix};

use crate::Activation;

#[derive(Debug, Clone)]
pub struct Layer {
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) gene_len: usize,
    pub(crate) activation: Activation,
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    act: Matrix<f64>, // activations
    act_func: fn(f64) -> f64,
    net: Matrix<f64>,
}

impl Layer {
    /// Create a new Layer with given input and output length and random weight and biases
    pub fn new(input_len: usize, output_len: usize, activation: Activation) -> Self {
        let weights = NewMatrix::from_shape_vec(
            (output_len,
            input_len),
            rand_vec_uniform(input_len * output_len),
        ).unwrap();
        let biases = NewMatrix::from_shape_vec((output_len, 1), rand_vec_uniform(output_len)).unwrap();
        let act_func = activation.get_func();
        Self {
            input_len,
            output_len,
            activation,
            gene_len: output_len * input_len + output_len,
            weights,
            biases,
            act: NewMatrix::from_shape_vec((input_len, 1), vec![0.0; input_len]).unwrap(),
            act_func,
            net: NewMatrix::from_shape_vec((output_len, 1), vec![0.0; output_len]).unwrap(),
        }
    }

    /// map the weight and biases in Matrices to flat vector
    pub fn genes(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        out.append(&mut self.weights.clone().into_raw_vec());
        out.append(&mut self.biases.clone().into_raw_vec());
        out
    }

    /// Return the number of genes in this layer
    pub fn num_genes(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Forward values through one layer
    pub(crate) fn forward(&mut self, m: &Matrix<f64>) -> Matrix<f64> {
        println!("\n\nFORWARD CALLED");
        println!("\n--{:?}---\n", self);
        let net = &self.weights * m + &self.biases;
        println!("\n--NET--{:?}---\n", net);
        //net.apply(&self.act_func)
        net.map(|x| (&self.act_func)(*x))
    }

    /// Set the weights for the layer
    pub(crate) fn set_weights(&mut self, w: Matrix<f64>) {
        assert_eq!(self.weights.nrows(), w.nrows());
        assert_eq!(self.weights.ncols(), w.ncols());
        self.weights = w;
    }

    /// Set the biases for the layer
    pub(crate) fn set_biases(&mut self, b: Matrix<f64>) {
        assert_eq!(self.biases.nrows(), b.nrows());
        assert_eq!(self.biases.ncols(), b.ncols());
        self.biases = b;
    }

    // set weights and biases of layer to the supplied genes
    // panics if genes.len() is wrong
    pub fn set_genes(&mut self, genes: &Vec<f64>) {
        let w_end = self.output_len * self.input_len;
        let weights: Matrix<f64> =
        NewMatrix::from_shape_vec((self.output_len, self.input_len), genes[..w_end].to_vec()).unwrap();
        self.set_weights(weights);
        let biases: Matrix<f64> = NewMatrix::from_shape_vec((self.output_len, 1), genes[w_end..].to_vec()).unwrap();
        self.set_biases(biases);
    }
}

/// Generate a r/andom vector of given length using a uniform distribution
/// values in range [-1.0, 1.0]
fn rand_vec_uniform(length: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..length).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()
    // let mut out: Vec<f64> = vec![0.0; length];
    // for i in 0..length {
    //     out[i] = ;
    // }
    // out
}

#[cfg(test)]
mod tests {
    extern crate round;

    use super::*;
    use round::*;

    #[test]
    fn layer_forward1() {
        let mut l = Layer::new(3, 1, Activation::Relu);
        println!("\n\n--L--{:?}---\n", l.weights);
        let w = NewMatrix::from_shape_vec((1, 3), vec![0.2, 0.4, 0.8]).unwrap();
        println!("\n\n--W--{:?}---\n", w);
        l.set_weights(w);
        println!("\n\n--L--{:?}---\n", l.weights);
        let b = NewMatrix::from_shape_vec((1, 1), vec![0.0]).unwrap();
        println!("\n\n--B--{:?}---\n", b);
        println!("\n\n--L-before-{:?}---\n", l.biases);
        l.set_biases(b);
        println!("\n\n--L-after-{:?}---\n", l.biases);

        let input = NewMatrix::from_shape_vec((3, 1), vec![0.2, 0.4, 0.8]).unwrap();
        let output = l.forward(&input);

        println!("\n\n--OUTPUT--{:?}---\n\n", output);

        assert_eq!(output.ncols(), 1);
        assert_eq!(output.nrows(), 1);
        assert_eq!(round(output.row(0)[0], 2), 0.84);
    }

    #[test]
    fn layer_forward2() {
        let mut l = Layer::new(3, 3, Activation::Relu);
        println!("\n\n--L--{:?}---\n", l);
        let w = NewMatrix::from_shape_vec((3, 3), vec![1.0; 9]).unwrap();
        println!("\n\n--W2--{:?}---\n", w);
        l.set_weights(w);
        println!("\n\n--L--{:?}---\n", l.weights);
        let b = NewMatrix::from_shape_vec((3, 1), vec![0.0; 3]).unwrap();
        l.set_biases(b);
        println!("\n\n--L2--{:?}---\n", l);

        let input = NewMatrix::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
        let output = l.forward(&input);

        println!("\n--output: {:?}", output);
        assert_eq!(output, NewMatrix::from_shape_vec((3, 1), vec![3.0, 3.0, 3.0]).unwrap());
    }

    #[test]
    fn layer_set_weights1() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let w: Matrix<f64> = NewMatrix::from_shape_vec((1, 3), vec![1.0; 3]).unwrap();
        l.set_weights(w);
    }

    #[test]
    #[should_panic]
    fn layer_set_weights2() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let w: Matrix<f64> = NewMatrix::from_shape_vec((3, 4), vec![1.0; 3]).unwrap();
        l.set_weights(w);
    }

    #[test]
    fn layer_set_biases() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let b: Matrix<f64> = NewMatrix::from_shape_vec((1, 1), vec![0.0]).unwrap();
        l.set_biases(b);
    }

    #[test]
    fn layer_enc_fit() {
        // TODO: test layer_enc_fit
        // let mut l = Layer::new(3, 1, Activation::Relu);
        //
        // // set weights and biases of layer
        // let w: Matrix<f64> = Matrix::new(1, 3, vec![1.0; 3]);
        // l.set_weights(w);
        // let b: Matrix<f64> = Matrix::new(1, 1, vec![1.0]);
        // l.set_biases(b);
        //
        // // do forward pass so that derived results are set
        // let inputs: Matrix<f64> = Matrix::new(3, 1, vec![1.0;3]);
        // l.forward(&inputs);
        //
        // let fit = 0.1;
        // let enc_fit = l.enc_fit(fit);
        //
        // assert_eq!(enc_fit, vec![0.025])
    }

    #[test]
    fn layer_genes() {
        let l = Layer::new(3, 1, Activation::Relu);

        let genes = l.genes();
        assert_eq!(genes.len(), 4);
    }

    #[test]
    fn layer_gene_len() {
        let l = Layer::new(3, 1, Activation::Relu);

        assert_eq!(l.gene_len, 4);

        let l = Layer::new(3, 3, Activation::Relu);

        assert_eq!(l.gene_len, 12);
    }

    #[test]
    fn layer_set_genes() {
        let mut l = Layer::new(3, 1, Activation::Relu);
        let genes: Vec<f64> = vec![1.0; 4];
        l.set_genes(&genes);
        assert_eq!(l.genes(), genes);

        let mut l = Layer::new(3, 3, Activation::Relu);
        let genes: Vec<f64> = vec![1.0; 12];
        l.set_genes(&genes);
        assert_eq!(l.genes(), genes);
    }
}
