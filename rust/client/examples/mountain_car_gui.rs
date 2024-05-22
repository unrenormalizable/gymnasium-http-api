extern crate chrono;
extern crate gymnasium;
extern crate serde_json;

use gymnasium::*;
use serde_json::to_value;
use std::rc::Rc;

fn main() -> ui::Result {
    let env = Environment::<BoxSpace<Continous>, BoxSpace<Continous>>::new(
        "http://127.0.0.1:40004",
        "MountainCarContinuous-v0",
        None,
        None,
        None,
        &[("render_mode", to_value("rgb_array").unwrap())],
    )
    .rc();
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
