mod environments;
mod math;
mod mdps;

use environments::gym_adapter::*;
use gymnasium_rust_client::*;
use mdps::mdp::*;
use mdps::mdp_solver::*;
use mdps::solvers::policy_iteration::*;
use serde_json::{to_value, Value};
use std::collections::HashMap;

fn main() {
    let gc = GymClient::new("http://localhost", 40004);

    let kwargs = HashMap::<&str, Value>::from([
        ("render_mode", to_value("ansi").unwrap()),
        //("map_name", to_value("8x8").unwrap()),
        ("is_slippery", to_value(true).unwrap()),
        //(
        //    "desc",
        //    to_value(["SFFFFHHH", "FFFFFHHH", "HHHFFHHH", "HHHFGHHH"]).unwrap(),
        //),
    ]);
    let env = gc.make_env("FrozenLake-v1", None, None, None, kwargs);
    let mdp = &GymAdapter::new(&env, 0.9) as &dyn Mdp;
    let theta = 1e-8;
    let mut p = PolicyIteration::new(mdp, 0., 0);
    let ret = p.exec(theta, None);
    println!(
        "Theta: {}, Policy stable: {}, Number of iterations: {}",
        theta, ret.0, ret.1
    );

    let v_star = (0..mdp.n_s())
        .map(|s| p.v_star(s as Discrete))
        .collect::<Vec<_>>();
    println!("{v_star:?}");
    let pi_star = (0..mdp.n_s())
        .map(|s| p.pi_star(s as Discrete))
        .collect::<Vec<_>>();
    println!("{pi_star:?}");
    let mut q_star = Vec::new();
    for s in 0..mdp.n_s() {
        for a in 0..mdp.n_a() {
            q_star.push(p.q_star(s as Discrete, a as Discrete))
        }
    }
    println!("{q_star:?}");

    let mut total_reward = 0.;
    let mut curr_state = env.reset(None)[0].discrete_value();

    for i in 0.. {
        let curr_action = p.pi_star(curr_state).unwrap();
        let step_info = env.step(&[ObsActSpaceItem::Discrete(curr_action)]);
        let render_frame = env.render();
        print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
        println!("{}", render_frame.as_str().unwrap());
        assert_eq!(step_info.observation.len(), env.action_space_sample().len());
        total_reward += step_info.reward;

        let next_state = step_info.observation[0].discrete_value();
        println!("Info: {i}\n  Action taken : {curr_action}\n  Current state: {curr_state}\n  Next state   : {next_state}\n  Total rewards: {total_reward}");
        if step_info.truncated || step_info.terminated {
            break;
        }
        curr_state = next_state;
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    std::thread::sleep(std::time::Duration::from_millis(100000));
}
