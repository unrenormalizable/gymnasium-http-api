extern crate chrono;
extern crate gymnasium;
extern crate serde_json;

use gymnasium::{mdps::*, *};
use serde_json::to_value;
use std::rc::Rc;

// NOTE: Replace the env_name in this sample to test out various environments in GUI.

fn main() -> ui::Result {
    let env = Rc::new(Environment::new(
        "http://127.0.0.1:40004",
        "CliffWalking-v0",
        None,
        None,
        None,
        &[("render_mode", to_value("rgb_array").unwrap())],
    ));
    let policy = RandomEnvironmentPolicy {
        env: Rc::clone(&env),
    };

    ui::GymnasiumApp::run(
        env.client_base_url(),
        env.instance_id(),
        None,
        Rc::new(policy),
    )
}
