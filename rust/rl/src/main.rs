mod environments;
mod math;
mod mdps;

use environments::gym_adapter::*;
use gymnasium::*;
use mdps::{
    mdp::*, mdp_simulator::*, mdp_solver::*, mdp_solver_policy::*, solvers::mc_methods::*,
    solvers::*,
};
use serde_json::to_value;
use std::rc::Rc;

fn main() -> ui::Result {
    let env = Rc::new(Environment::new(
        "http://127.0.0.1:40004",
        "FrozenLake-v1",
        None,
        None,
        None,
        &[
            ("is_slippery", to_value(true).unwrap()),
            ("render_mode", to_value("rgb_array").unwrap()),
            ("desc", to_value(["SFFF", "FFFF", "FFFF", "HFFG"]).unwrap()),
        ],
    ));

    //let ga = Rc::new(GymAdapter::new(Rc::clone(&env), 0.9));
    //let mdp = ga as Rc<dyn Mdp>;
    //let theta = 1e-8;
    //let vi = &mut policy_iteration::PolicyIteration::new(Rc::clone(&mdp), 0., 0);
    //let _ret = vi.exec(theta, Some(1));

    //let v_star = (0..mdp.n_s())
    //    .map(|s| vi.v_star(s as Discrete))
    //    .collect::<Vec<_>>();
    //println!("{v_star:?}");
    //let pi_star = (0..mdp.n_s())
    //    .map(|s| p.pi_star(s as Discrete))
    //    .collect::<Vec<_>>();
    //println!("{pi_star:?}");
    //let mut q_star = Vec::new();
    //for s in 0..mdp.n_s() {
    //    for a in 0..mdp.n_a() {
    //        q_star.push(p.q_star(s as Discrete, a as Discrete))
    //    }
    //}
    //println!("{q_star:?}");

    //Ok(())

    ////let x = env.episode_samples(10000, Some(2178));
    ////println!("{}", x.len());
    let ga = Rc::new(GymAdapter::new(Rc::clone(&env), 0.9));
    let mdp = ga as Rc<dyn Mdp>;
    let theta = 1e-8;
    let pi = &mut policy_iteration::PolicyIteration::new(Rc::clone(&mdp), 0., 0);
    let ret = pi.exec(theta, None);
    println!(
        "Theta: {}, Policy stable: {}, Number of iterations: {}",
        theta, ret.0, ret.1
    );

    //let pi = &*pi;
    //let v_star = (0..mdp.n_s())
    //    .map(|s| pi.v_star(s as Discrete))
    //    .collect::<Vec<_>>();
    //println!("{v_star:?}");
    ////let pi_star = (0..mdp.n_s())
    ////    .map(|s| pi.pi_star(s as Discrete))
    ////    .collect::<Vec<_>>();
    ////println!("{pi_star:?}");
    ////let mut q_star = Vec::new();
    ////for s in 0..mdp.n_s() {
    ////    for a in 0..mdp.n_a() {
    ////        q_star.push(pi.q_star(s as Discrete, a as Discrete))
    ////    }
    ////}
    ////println!("{q_star:?}");

    //let ep_gen = Rc::clone(&env) as Rc<dyn EpisodeGenerator>;
    //let v = mc_first_visit(ep_gen, 1.0, 16, 10000);
    //println!("{v:?}");

    let solver = Rc::new(pi.clone()) as Rc<dyn MdpSolver<bool>>;
    let policy = MdpSolverPolicy { mdp_solver: solver };
    ui::GymnasiumApp::run(
        env.client_base_url(),
        env.instance_id(),
        None,
        Rc::new(policy),
    )
}

// TODO: Move this else where.
impl EpisodeGenerator for Environment {
    fn generate(&self, n: usize) -> Vec<Vec<EpisodeEvent>> {
        self.episode_samples(n, None)
    }
}
