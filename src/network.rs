use na::DMatrix as Matrix;

use crate::{Activation, Layer};

#[derive(Debug, Clone)]
/// Artificial Neural Network
pub struct ANN {
    num_inputs: usize,
    num_outputs: usize,
    pub(crate) layers: Vec<Layer>,
    num_genes: usize,
}

impl ANN {
    /// Create a new artificial neural network
    /// with a given number of inputs and outputs and an activation function
    pub fn new(num_inputs: usize, num_outputs: usize, act_func: Activation) -> ANN {
        let layers = vec![Layer::new(num_inputs, num_outputs, act_func)];
        let num_genes = layers.iter().map(|l| l.num_genes()).sum();

        ANN {
            num_inputs,
            num_outputs,
            layers,
            num_genes,
        }
    }

    /// Add a new hidden layer with a given neuron count and activation function.
    /// This modifies the previous and following layer to match io in each layer
    pub fn add_layer(&mut self, neuron_count: usize, act: Activation) {
        let last_layer_idx = self.layers.len() - 1;

        // set new layer as output layer
        self.layers
            .push(Layer::new(neuron_count, self.num_outputs, act));

        // modify previous layer output_len to match neuron_count of new layer
        let old_input_len = self.layers[last_layer_idx].input_len;
        let old_activation = self.layers[last_layer_idx].activation;
        self.layers[last_layer_idx] = Layer::new(old_input_len, neuron_count, old_activation);

        // re-compute num_genes
        self.num_genes = self.layers.iter().map(|l| l.num_genes()).sum();
    }

    /// forward the inputs through the network
    /// Returns an output of length self.num_outputs
    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut prev_output = Matrix::from_vec(self.num_inputs, 1, inputs);
        for l in 0..self.layers.len() {
            prev_output = self.layers[l].forward(&prev_output);
        }

        return prev_output.as_slice().into();
    }

    /// Return the number of genes in the network
    pub fn num_genes(&self) -> usize {
        self.num_genes
    }

    /// returns the genes representing the network
    pub fn genes(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for l in &self.layers {
            out.append(&mut l.genes())
        }
        out
    }

    /// update the network weights and biases with new genes
    pub(crate) fn set_genes(&mut self, genes: &Vec<f64>) {
        assert_eq!(genes.len(), self.num_genes());

        let mut start: usize = 0;
        for l in &mut self.layers {
            let end = start + l.gene_len;
            l.set_genes(&genes[start..end]);
            start += l.gene_len;
        }
    }

    /// randomize returns a new randomized instance of ANN
    pub(crate) fn randomize(&self) -> ANN {
        let mut layers: Vec<Layer> = Vec::new();
        for l in &self.layers {
            layers.push(Layer::new(l.input_len, l.output_len, l.activation))
        }
        let num_genes = layers.iter().map(|l| l.num_genes()).sum();
        ANN {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            layers,
            num_genes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_new() {
        let num_inputs: usize = 3;
        let num_outputs: usize = 1;
        let nn = ANN::new(num_inputs, num_outputs, Activation::Tanh);
        assert_eq!(nn.layers.len(), 1);
        assert_eq!(nn.num_inputs, num_inputs);
        assert_eq!(nn.num_outputs, num_outputs);
    }

    #[test]
    fn num_genes() {
        let nn = ANN::new(3, 1, Activation::Relu);
        assert_eq!(nn.num_genes(), 4);
    }

    #[test]
    fn network_forward() {
        let mut nn = ANN::new(3, 1, Activation::Relu);

        // set weights and biases for input layer
        let w: Matrix<f64> = Matrix::from_vec(1, 3, vec![1.0; 3]);
        nn.layers[0].set_weights(w);
        let b: Matrix<f64> = Matrix::from_vec(1, 1, vec![1.0]);
        nn.layers[0].set_biases(b);

        let input = vec![1.0; 3];
        let output = nn.forward(input);

        assert_eq!(
            Matrix::from_vec(1, 1, output),
            Matrix::from_vec(1, 1, vec![4.0])
        );
    }

    #[test]
    fn add_layer() {
        let mut nn = ANN::new(3, 1, Activation::Relu);

        nn.add_layer(3, Activation::Relu);

        // set weights and biases of input layer
        let w: Matrix<f64> = Matrix::from_vec(3, 3, vec![1.0; 9]);
        nn.layers[0].set_weights(w);
        let b: Matrix<f64> = Matrix::from_vec(3, 1, vec![0.0; 3]);
        nn.layers[0].set_biases(b);

        // set weights and biases for hidden layer
        let w: Matrix<f64> = Matrix::from_vec(1, 3, vec![1.0; 3]);
        nn.layers[1].set_weights(w);
        let b: Matrix<f64> = Matrix::from_vec(1, 1, vec![0.0]);
        nn.layers[1].set_biases(b);

        let input = vec![1.0; 3];
        let output = nn.forward(input);

        assert_eq!(
            Matrix::from_vec(1, 1, output),
            Matrix::from_vec(1, 1, vec![9.0])
        );
    }

    #[test]
    fn network_genes() {
        let mut nn = ANN::new(3, 1, Activation::Relu);

        let genes = nn.genes();

        assert_eq!(genes.len(), 4);

        nn.add_layer(3, Activation::Relu);
        let genes = nn.genes();

        assert_eq!(genes.len(), 16);
    }

    #[test]
    fn network_set_genes() {
        let mut nn = ANN::new(3, 1, Activation::Relu);
        let genes = vec![1.0; 4];
        nn.set_genes(&genes);

        nn.add_layer(3, Activation::Relu);

        let genes = vec![1.0; 16];
        nn.set_genes(&genes);
        assert_eq!(nn.genes(), genes);
    }
}
