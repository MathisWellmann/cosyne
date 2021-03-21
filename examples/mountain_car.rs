use cosyne::{Activation, Config, ANN, Environment, Cosyne, Genome};
use gym_rs::{MountainCarEnv, ActionType, GifRender, GymEnv};
use std::time::Instant;

fn main() {
    pretty_env_logger::init();

    let config = Config::new(100, 1);
    let env = Box::new(MountainCarEvaluator{} );
    let mut nn = ANN::new(2, 1, Activation::Tanh);
    nn.add_layer(4, Activation::Tanh);
    let mut cosyne = Cosyne::new(env, nn, config);
    let t0 = Instant::now();
    for _ in 0..100 {
        cosyne.step();
    }
    let champion = cosyne.champion();
    println!("champion: {:?}", champion);
    println!("training time: {}s", t0.elapsed().as_secs());

    let filename = "img/mountain_car_fitness_history.png";
    cosyne.plot_fitness_history(filename, (1920, 1080)).unwrap();

    render_champion(&mut champion.clone());
}

struct MountainCarEvaluator {}

impl Environment for MountainCarEvaluator {
    fn evaluate(&self, nn: &mut ANN) -> f64 {
        let mut env = MountainCarEnv::default();

        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_rewards: f64 = 0.0;
        let mut steps: usize = 0;
        while !end {
            if steps > 500 {
                break;
            }
            let output = nn.forward(state);
            let action = ActionType::Continuous(vec![output[0] * 2.0]);
            let (s, reward, done, _info) = env.step(action);
            end = done;
            state = s;
            total_rewards += (3.0 * state[0]).sin() * 0.45 + 0.55;
            steps += 1;
        }
        total_rewards
    }
}

fn render_champion(champion: &mut Genome) {
    println!("rendering champion...");

    let mut env = MountainCarEnv::default();

    let mut render = GifRender::new(
        540,
        540,
        "img/mountain_car_champion.gif",
        50
    ).unwrap();

    let mut state: Vec<f64> = env.reset();

    let mut end: bool = false;
    let mut steps: usize = 0;
    while !end {
        if steps > 500 {
            break;
        }
        let output = champion.network.forward(state);
        let action = ActionType::Continuous(vec![output[0] * 2.0]);
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}