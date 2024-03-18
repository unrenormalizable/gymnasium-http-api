extern crate gymnasium;

use gymnasium::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = GymClient::new("http://localhost", 5000);

    let envs = c.get_envs()?;
    println!("Open environments: {:?}", envs);

    let env = c.make_env("FrozenLake-v1", Some("ansi"))?;

    println!("observation space:\n{:?}\n", env.observation_space());
    println!("action space:\n{:?}\n", env.action_space());
    println!("transtion[0][1]:\n{:?}\n", env.get_transitions(0, 1));
    println!("transtion[15][3]:\n{:?}\n", env.get_transitions(15, 3));

    for ep in 0..100 {
        let _ = env.reset(Some(2718));
        let mut tot_reward = 0.;
        loop {
            let action = env.action_space().sample();
            let state = env.step(action).unwrap();
            let render_frame = env.render()?;
            print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
            println!("{}", render_frame);
            assert_eq!(
                state.observation.len(),
                env.observation_space().sample().len()
            );
            tot_reward += state.reward;

            if state.truncated || state.terminated {
                break;
            }
        }
        println!("Finished episode {} with total reward {}", ep, tot_reward);
    }

    Ok(())
}
