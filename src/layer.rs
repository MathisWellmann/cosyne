use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use rand::{thread_rng, Rng};
use crate::activations::{Activation, relu, soft_plus, soft_sign, tanh};

#[derive(Debug, Clone)]
pub struct Layer {
    input_len: usize,
    output_len: usize,
    activation: Activation,
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    act: Matrix<f64>,  // activations
    act_func: fn(f64) -> f64,
}

impl Layer {
    // new returns a new Layer with given input and output length
    pub fn new(input_len: usize, output_len: usize, activation: Activation) -> Layer {
        let weights = Matrix::new(output_len, input_len, rand_vec(input_len * output_len));
        let biases = Matrix::new(output_len, 1, rand_vec(output_len));
        let act_func = match activation{
            Activation::Relu => relu,
            Activation::SoftPlus => soft_plus,
            Activation::SoftSign => soft_sign,
            Activation::Tanh => tanh,
        };
        return Layer{
            input_len,
            output_len,
            activation,
            weights,
            biases,
            act: Matrix::new(input_len, 1, vec![0.0; input_len]),
            act_func,
        }
    }

    // genes maps the weight and biases in Matrices to flat vector
    pub fn genes(&self) -> Vec<f64> {
        // TODO:
        return Vec::new()
    }

    // enc_fit encodes the fitness to their respective weights and biases
    pub fn enc_fit(&mut self, fit: f64) {
        // TODO:
    }

    pub fn forward(&self, m: &Matrix<f64>) -> Matrix<f64> {
        let res = &self.weights * m + &self.biases;
        return res.apply(&self.act_func);
    }

    pub fn set_weights(&mut self, w: Matrix<f64>) {
        assert_eq!(self.weights.rows(), w.rows());
        assert_eq!(self.weights.cols(), w.cols());
        self.weights = w;
    }

    pub fn set_biases(&mut self, b: Matrix<f64>) {
        assert_eq!(self.biases.rows(), b.rows());
        assert_eq!(self.biases.cols(), b.cols());
        self.biases = b;
    }

    pub fn input_len(&self) -> usize {
        return self.input_len
    }

    pub fn output_len(&self) -> usize {
        return self.output_len
    }

    pub fn activation(&self) -> Activation {
        return self.activation.clone()
    }
}

// generate a random vector of given length
fn rand_vec(length: usize) -> Vec<f64> {
    let mut out: Vec<f64> = vec![0.0; length];
    let mut rng = thread_rng();
    for i in 0..length {
        out[i] = rng.gen::<f64>();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_forward1() {
        let mut l = Layer::new(3, 1, Activation::Relu);
        let w = Matrix::new(1, 3, vec![0.2, 0.4, 0.8]);
        l.set_weights(w);
        let b = Matrix::new(1, 1, vec![0.0]);
        l.set_biases(b);
        let input = Matrix::new(3, 1, vec![0.2, 0.4, 0.8]);
        let output = l.forward(&input);

        assert_eq!(output.cols(), 1);
        assert_eq!(output.rows(), 1);
        assert_eq!(output.get_row(0).unwrap()[0], 0.84);
    }

    #[test]
    fn layer_forward2() {
        let mut l = Layer::new(3, 3, Activation::Relu);
        let w = Matrix::new(3, 3, vec![1.0; 9]);
        l.set_weights(w);
        let b = Matrix::new(3, 1, vec![0.0; 3]);
        l.set_biases(b);

        let input = Matrix::new(3, 1, vec![1.0; 3]);
        let output = l.forward(&input);

        println!("output: {:?}", output.get_row(0).unwrap());
        assert_eq!(output, Matrix::new(3, 1, vec![3.0, 3.0, 3.0]));
    }

    #[test]
    fn set_weights1() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let w: Matrix<f64> = Matrix::new(1, 3, vec![1.0; 3]);
        l.set_weights(w);
    }

    #[test]
    #[should_panic]
    fn set_weights2() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let w: Matrix<f64> = Matrix::new(3, 4, vec![1.0; 3]);
        l.set_weights(w);
    }

    #[test]
    fn set_biases() {
        let mut l = Layer::new(3, 1, Activation::Relu);

        let b: Matrix<f64> = Matrix::new(1, 1, vec![0.0]);
        l.set_biases(b);
    }
}