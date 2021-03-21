use cosyne::{Activation, Config, Cosyne, Environment, ANN};

#[test]
fn xor() {
    let config = Config::new(100, 5);
    let env = Box::new(XorEnvironment {});
    let activation = Activation::Relu;
    let mut nn = ANN::new(2, 1, activation);
    nn.add_layer(2, Activation::Relu);
    let mut cosyne = Cosyne::new(env, nn, config);
    for _ in 0..100 {
        cosyne.step();
    }
    let champion = cosyne.champion();
    println!("champion: {:?}", champion);
    assert!(champion.fitness > 3.9);
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
