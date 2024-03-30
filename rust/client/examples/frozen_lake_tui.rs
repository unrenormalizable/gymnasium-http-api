extern crate gymnasium;
extern crate serde_json;

use gymnasium::*;
use serde_json::to_value;

fn main() {
    let envs = Environment::envs("http://127.0.0.1:40004");
    println!("Open environments: {:?}", envs);
    let env = Environment::new(
        "http://127.0.0.1:40004",
        "FrozenLake-v1",
        Some(100),
        Some(false),
        Some(true),
        &[
            ("render_mode", to_value("ansi").unwrap()),
            ("map_name", to_value("8x8").unwrap()),
            ("is_slippery", to_value(true).unwrap()),
            //("desc", to_value(&["SHHH", "FHHH", "FHHF", "FFFG"])?),
        ],
    );

    println!("observation space:\n{:?}\n", env.observation_space());
    println!("action space:\n{:?}\n", env.action_space());
    let transitions_0_0 = &env.transitions()[&(14, 2)];
    println!("transtion:\n{:?}\n", transitions_0_0);

    env.reset(Some(2718));

    for ep in 0..100 {
        let _ = env.reset(Some(2718));
        let mut tot_reward = 0.;
        loop {
            let action = env.action_space_sample();
            let state = env.step(&action);
            let render_frame = env.render();
            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
            println!("{}", render_frame.as_str().unwrap());
            tot_reward += state.reward;

            if state.truncated || state.terminated {
                break;
            }
        }
        println!("Finished episode {} with total reward {}", ep, tot_reward);
    }
}
