use crate::layer::Layer;
use crate::activations::Activation;
use rulinalg::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct ANN {
    num_inputs: usize,
    num_outputs: usize,
    pub(crate) layers: Vec<Layer>,
}

impl ANN {
    pub fn new(num_inputs: usize, num_outputs: usize, act_func: Activation) -> ANN {
        let mut layers: Vec<Layer> = Vec::new();
        layers.push(Layer::new(num_inputs, num_outputs, act_func.clone()));
        return ANN{
            num_inputs,
            num_outputs,
            layers,
        }
    }

    // add a new hidden layer. this modifies the last and second to last layer to match neuron count
    pub fn add_layer(&mut self, neuron_count: usize, act: Activation) {
        let last = self.layers.len() - 1;
        // set new layer as output layer
        self.layers.push(Layer::new(neuron_count, self.num_outputs, act));

        // modify previous layer output_len to match neuron_count of new layer
        let old_input_len = self.layers[last].input_len();
        let old_activation = self.layers[last].activation();
        self.layers[last] = Layer::new(old_input_len, neuron_count, old_activation);
    }

    // forward the inputs through the network
    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut prev_output = Matrix::new(self.num_inputs, 1, inputs);
        for l in 0..self.layers.len() {
            prev_output = self.layers[l].forward(&prev_output);
        }

        return prev_output.into_vec()
    }

    pub fn enc_fitness(&mut self, _fit: f64) -> Vec<f64> {
        // TODO: enc_fitness
        vec![1.0; self.genes().len()]
    }

    // returns the genes representing the network
    pub fn genes(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for l in &self.layers {
            out.append(&mut l.genes())
        }
        out
    }

    /// Return the number of genes in the network
    pub fn num_genes(&self) -> usize {
        self.layers.iter().map(|l| l.num_genes()).count()
    }

    // randomize returns a new randomized instance of ANN
    pub fn randomize(&self) -> ANN {
        let mut layers: Vec<Layer> = Vec::new();
        for l in &self.layers {
            layers.push(Layer::new(l.input_len(), l.output_len(), l.activation()))
        }
        return ANN {
            num_inputs: self.num_inputs,
            num_outputs: self.num_outputs,
            layers
        }
    }

    // update the network weights and biases with new genes
    pub fn set_genes(&mut self, genes: &Vec<f64>) {
        // assert_eq!(genes.len(), self.num_genes());

        let mut start: usize = 0;
        for l in &mut self.layers {
            let end = start + l.gene_len();
            l.set_genes(&genes[start..end].to_vec());
            start += l.gene_len();
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use rulinalg::matrix::Matrix;

    #[test]
    fn network_forward() {
        let mut nn = ANN::new(3, 1, Activation::Relu);

        // set weights and biases for input layer
        let w: Matrix<f64> = Matrix::new(1, 3, vec![1.0; 3]);
        nn.layers[0].set_weights(w);
        let b: Matrix<f64> = Matrix::new(1, 1, vec![1.0]);
        nn.layers[0].set_biases(b);


        let input = vec![1.0; 3];
        let output = nn.forward(input);

        assert_eq!(Matrix::new(1, 1, output), Matrix::new(1, 1, vec![4.0]));
    }

    #[test]
    fn add_layer() {
        let mut nn = ANN::new(3, 1, Activation::Relu);

        nn.add_layer(3, Activation::Relu);

        // set weights and biases of input layer
        let w: Matrix<f64> = Matrix::new(3, 3, vec![1.0; 9]);
        nn.layers[0].set_weights(w);
        let b: Matrix<f64> = Matrix::new(3, 1, vec![0.0; 3]);
        nn.layers[0].set_biases(b);

        // set weights and biases for hidden layer
        let w: Matrix<f64> = Matrix::new(1, 3, vec![1.0; 3]);
        nn.layers[1].set_weights(w);
        let b: Matrix<f64> = Matrix::new(1, 1, vec![0.0]);
        nn.layers[1].set_biases(b);

        let input = vec![1.0; 3];
        let output = nn.forward(input);

        assert_eq!(Matrix::new(1, 1, output), Matrix::new(1, 1, vec![9.0]));
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
        nn.add_layer(3, Activation::Relu);

        let genes = vec![1.0; 16];
        nn.set_genes(&genes);
        assert_eq!(nn.genes(), genes);
    }
}