use cosyne::environment::Environment;
use cosyne::network::ANN;
use cosyne::activations::Activation;
use cosyne::population::Population;
use std::time::{Instant, Duration};

fn main() {
    let pop_size = 100;
    let input_len = 2;
    let output_len = 2;
    let activation = Activation::Relu;
    let env = Box::new(InvertedPendulum::new());
    let mut nn = ANN::new(input_len, output_len, activation.clone());
    nn.add_layer(input_len, activation);
    let mut pop = Population::new(env, pop_size, nn);

    for g in 0..100 {
        let best = pop.generation();
        println!("gen {} best fitness: {:?}", g, best.fitness);
    }
}

struct InvertedPendulum{
    g: f64,
    cart: Cart,
    pendulum: Pendulum,
    simulation_time: u64,
}

impl InvertedPendulum {
    pub fn new() -> InvertedPendulum {
        return InvertedPendulum{
            g: 9.81,
            cart: Cart::new(500.0, 5.0),
            pendulum: Pendulum::new(1.0, 1.0, 1.0),
            simulation_time: 35,
        }
    }

    pub fn apply_control_input(&mut self, force: f64, x_t_m2: f64, theta_dot: f64, theta_t_m2: f64) {
        let c = &self.cart;
        let p = &self.pendulum;

        let theta_sin = p.theta.sin();
        let theta_cos = p.theta.cos();
        let theta_double_dot = ((c.mass + p.ball_mass) * self.g * theta_sin)
            + (force * theta_cos)
            - (p.ball_mass * theta_dot.powi(2) * p.length * theta_sin * theta_cos)
            / (p.length * (c.mass + (p.ball_mass * theta_sin.powi(2))));
        let x_double_pot = ((p.ball_mass * self.g * theta_sin * theta_cos)
            - (p.ball_mass * p.length * theta_sin * theta_dot.powi(2))
            + force / (c.mass + p.ball_mass * theta_sin.powi(2)));
        self.cart.x += x_double_pot
            + (c.x - x_t_m2);
        self.pendulum.theta += theta_double_dot
            + p.theta - theta_t_m2;
    }
}

impl Environment for InvertedPendulum {
    fn evaluate(&mut self, nn: &mut ANN) -> f64 {
        let mut fitness: f64 = 0.0;
        let mut theta_dot = 0.0;
        let mut theta_t_m1 = self.pendulum.theta;
        let mut theta_t_m2 = self.pendulum.theta;
        let mut x_t_m1 = self.cart.x;
        let mut x_t_m2 = self.cart.x;
        let mut prev_err = self.pendulum.find_error();
        for i in 0..1000*self.simulation_time {  // runs on millisecond basis
            let err: f64 = self.pendulum.find_error();
            if i > 0 {
                theta_dot = (theta_t_m1 - theta_t_m2);
                let x_dot = (x_t_m1 - x_t_m2);

                // get control input from nn
                let inputs = vec![theta_dot, x_dot];
                let output = nn.forward(inputs);
                let force = output[0];

                self.apply_control_input(force, x_t_m2, theta_dot, theta_t_m2);
            }

            prev_err = err;
            theta_t_m2 = theta_t_m1;
            theta_t_m1 = self.pendulum.theta;
            x_t_m2 = x_t_m1;
            x_t_m1 = self.cart.x;
        }

        return fitness
    }

    fn reset(&mut self) {
        self.cart = Cart::new(500.0, 5.0);
        self.pendulum = Pendulum::new(1.0, 1.0, 1.0);
    }
}

struct Cart{
    x: f64,
    mass: f64,
}

impl Cart {
    pub fn new(x: f64, mass: f64) -> Cart {
        return Cart{
            x,
            mass,
        }
    }
}

struct Pendulum{
    length: f64,
    theta: f64,
    ball_mass: f64,
}

impl Pendulum {
    pub fn new(length: f64, theta: f64, ball_mass: f64) -> Pendulum {
        return Pendulum{
            length,
            theta,
            ball_mass
        }
    }

    // find_error returns the error of the current pendulum position
    pub fn find_error(&self) -> f64 {
        let mut prev_err = (self.theta % (2.0 * std::f64::consts::PI)) - 0.0;
        if prev_err > std::f64::consts::PI {
            prev_err = prev_err - (2.0 * std::f64::consts::PI);
        }
        return prev_err
    }
}