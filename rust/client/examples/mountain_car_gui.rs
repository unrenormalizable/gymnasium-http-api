extern crate chrono;
extern crate gymnasium;
extern crate serde_json;

use gymnasium::*;
use serde_json::to_value;

fn main() -> ui::Result {
    let env = Environment::new(
        "http://127.0.0.1:40004",
        "MountainCar-v0",
        None,
        None,
        None,
        &[("render_mode", to_value("rgb_array").unwrap())],
    );
    let base_url = env.client_base_url().to_string();
    let instance_id = env.instance_id().to_string();
    let policy = policy::RandomEnvironmentPolicy { env: Box::new(env) };

    ui::GymnasiumApp::run(&base_url, &instance_id, None, Box::new(policy))
}
