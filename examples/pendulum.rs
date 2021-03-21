use cosyne::{Config, ANN, Activation, Cosyne, Genome, Environment};
use std::time::Instant;
use gym_rs::{PendulumEnv, GifRender, GymEnv, ActionType};

fn main() {
    pretty_env_logger::init();

    let config = Config::new(100, 1);
    let env = Box::new(PendulumEvaluator{} );
    let mut nn = ANN::new(3, 1, Activation::Tanh);
    nn.add_layer(3, Activation::Tanh);
    let mut cosyne = Cosyne::new(env, nn, config);
    let t0 = Instant::now();
    for _ in 0..100 {
        cosyne.step();
    }
    let champion = cosyne.champion();
    println!("champion: {:?}", champion);
    println!("training time: {}s", t0.elapsed().as_secs());

    let filename = "img/pendulum_fitness_history.png";
    cosyne.plot_fitness_history(filename, (1920, 1080)).unwrap();

    render_champion(&mut champion.clone());
}

struct PendulumEvaluator {}

impl Environment for PendulumEvaluator {
    fn evaluate(&self, nn: &mut ANN) -> f64 {
        let mut env = PendulumEnv::default();

        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f64 = 0.0;
        let mut steps: usize = 0;
        while !end {
            if steps > 150 {
                break;
            }
            let output = nn.forward(state);
            let action = ActionType::Continuous(vec![output[0] * 2.5]);
            let (s, reward, done, _info) = env.step(action);
            end = done;
            state = s;
            total_reward += reward;
            steps += 1;
        }
        total_reward
    }
}

fn render_champion(champion: &mut Genome) {
    println!("rendering champion...");

    let mut env = PendulumEnv::default();

    let mut render = GifRender::new(
        540,
        540,
        "img/pendulum_champion.gif",
        50,
    ).unwrap();

    let mut state: Vec<f64> = env.reset();

    let mut end: bool = false;
    let mut steps: usize = 0;
    while !end {
        if steps > 150 {
            break
        }
        let output = champion.network.forward(state);
        let action = ActionType::Continuous(vec![output[0] * 2.5]);
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}