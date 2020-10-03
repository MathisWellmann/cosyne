use cge::Activation;

use cosyne::{ Config, ANN, Cosyne, Environment};

fn main() {
    let config = Config::new_fixed_activation(100, Activation::Relu);
    let env = Box::new(XorEnvironment{});
    let activation = Activation::Relu;
    let mut nn = ANN::new(2, 1, activation);
    let mut cosyne = Cosyne::new(env, nn, config);
    let champion = cosyne.optimize(100);
    println!("champion: {:?}", champion);
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
}