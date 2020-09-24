use cosyne::population::Population;
use cosyne::environment::Environment;
use cosyne::network::ANN;
use cosyne::activations::Activation;

fn main() {
    let pop_size = 20;
    let input_len = 2;
    let output_len = 2;
    let activation = Activation::Relu;
    let env = Box::new(XorEnvironment{});
    let mut nn = ANN::new(input_len, output_len, activation);
    let mut pop = Population::new(env, pop_size, nn);

    for g in 0..10 {
        let best = pop.generation();
        println!("gen {} best fitness: {:?}", g, best.fitness);
    }

}

struct XorEnvironment {}

impl Environment for XorEnvironment {

    fn evaluate(&self, nn: &mut ANN) -> f64 {
        let mut distance: f64;

        let mut output = nn.forward(vec![0.0, 0.0]);
        distance = (0.0 - output[0]).abs();
        output = nn.forward(vec![0.0, 1.0]);
        distance += (1.0 - output[0]).abs();
        output = nn.forward(vec![1.0, 0.0]);
        distance += (1.0 - output[0]).abs();
        output = nn.forward(vec![1.0, 1.0]);
        distance += (0.0 - output[0]).abs();

        (4.0 - distance).powi(2)
    }

    fn reset(&mut self) {

    }
}