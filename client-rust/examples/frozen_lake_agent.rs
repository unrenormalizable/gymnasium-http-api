extern crate gymnasium_rust_client;
extern crate serde_json;

use gymnasium_rust_client::*;
use serde_json::{to_value, Value};
use std::collections::HashMap;

fn main() {
    let c = GymClient::new("http://localhost", 4000);

    let envs = c.get_envs();
    println!("Open environments: {:?}", envs);
    let kwargs = HashMap::<&str, Value>::from([
        ("render_mode", to_value("ansi").unwrap()),
        ("map_name", to_value("8x8").unwrap()),
        ("is_slippery", to_value(true).unwrap()),
        //("desc", to_value(&["SHHH", "FHHH", "FHHF", "FFFG"])?),
    ]);
    let env = c.make_env("FrozenLake-v1", Some(100), Some(false), Some(true), kwargs);

    let xxx = env.action_space_sample();
    println!("{xxx:?}");

    println!("observation space:\n{:?}\n", env.observation_space());
    println!("action space:\n{:?}\n", env.action_space());
    let transitions_0_0 = &env.transitions()[&(14, 2)];
    println!("transtion:\n{:?}\n", transitions_0_0);

    let x = env.reset(Some(2718));
    println!(">>>> {x:?}");

    for ep in 0..100 {
        let _ = env.reset(Some(2718));
        let mut tot_reward = 0.;
        loop {
            let action = env.action_space_sample();
            let state = env.step(&action);
            let render_frame = env.render();
            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
            println!("{}", render_frame);
            tot_reward += state.reward;

            if state.truncated || state.terminated {
                break;
            }
        }
        println!("Finished episode {} with total reward {}", ep, tot_reward);
    }
}
