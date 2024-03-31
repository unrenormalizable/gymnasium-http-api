use crate::Mdp;
use gymnasium::*;
use std::rc::Rc;

pub struct GymAdapter {
    name: String,
    env: Rc<Environment>,
    gamma: f32,
    transitions: Rc<Transitions>,
}

impl GymAdapter {
    pub fn new(env: Rc<Environment>, gamma: f32) -> Self {
        let transitions = env.transitions();

        Self {
            name: env.name().to_string(),
            env,
            gamma,
            transitions,
        }
    }
}

impl Mdp for GymAdapter {
    fn n_s(&self) -> usize {
        if let ObsActSpace::Discrete { n } = self.env.observation_space() {
            *n as usize
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    fn n_a(&self) -> usize {
        if let ObsActSpace::Discrete { n } = self.env.action_space() {
            *n as usize
        } else {
            panic!("'{}' is not an MDP.", self.name)
        }
    }

    // TODO: Should the transitions out of end states be removed.
    fn transitions(&self) -> Rc<Transitions> {
        Rc::clone(&self.transitions)
    }

    fn gamma(&self) -> f32 {
        self.gamma
    }
}
