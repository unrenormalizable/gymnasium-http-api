extern crate gymnasium;

use gymnasium::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = GymClient::new("http://localhost", 5000);

    let envs = c.get_envs()?;
    println!("{:?}", envs);

    let env = c.make_env("CartPole-v1")?;
    println!("{:?}", env.instance_id());

    println!("observation space:\n{:?}\n", env.observation_space());
    println!("action space:\n{:?}\n", env.action_space());

    Ok(())
}
