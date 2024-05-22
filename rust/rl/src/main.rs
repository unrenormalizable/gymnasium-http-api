mod algos;
mod envs;
mod math;

use algos::model_based::mdp::{pi::*, *};
use envs::gym_adapter::*;
use gymnasium::*;
use serde_json::to_value;
use std::rc::Rc;

fn main() -> ui::Result {
    let env = Environment::new(
        "http://127.0.0.1:40004",
        "FrozenLake-v1",
        None,
        None,
        None,
        &[
            ("render_mode", to_value("rgb_array").unwrap()),
            ("map_name", to_value("8x8").unwrap()),
        ],
    )
    .rc();
    let base_url = env.client_base_url().to_string();
    let instance_id = env.instance_id().to_string();

    let ga = Rc::new(GymAdapter::new(Rc::clone(&env), 0.9));
    let mdp = ga as Rc<dyn Mdp>;
    let theta = 1e-8;
    let pi = &mut PolicyIteration::new(Rc::clone(&mdp), 0., 0);
    let ret = pi.exec(theta, None);
    println!(
        "Theta: {}, Policy stable: {}, Number of iterations: {}",
        theta, ret.0, ret.1
    );

    let pi = &*pi;
    let v_star = (0..mdp.n_s())
        .map(|s| pi.v_star(s as Discrete))
        .collect::<Vec<_>>();
    println!("{v_star:?}");
    let pi_star = (0..mdp.n_s())
        .map(|s| pi.pi_star(s as Discrete))
        .collect::<Vec<_>>();
    println!("{pi_star:?}");
    let mut q_star = Vec::new();
    for s in 0..mdp.n_s() {
        for a in 0..mdp.n_a() {
            q_star.push(pi.q_star(s as Discrete, a as Discrete))
        }
    }
    println!("{q_star:?}");

    let policy = Rc::new(pi.clone());
    ui::GymnasiumApp::run(&base_url, &instance_id, None, policy)
}
