use crate::Mdp;
use gymnasium::*;

pub struct GymAdapter<'a> {
    name: String,
    env: &'a Environment,
    gamma: f32,
    transitions: Transitions,
}

impl<'a> GymAdapter<'a> {
    pub fn new(env: &'a Environment, gamma: f32) -> Self {
        let transitions = env.transitions();

        Self {
            name: env.name().to_string(),
            env,
            gamma,
            transitions,
        }
    }
}

impl<'a> Mdp<'a> for GymAdapter<'a> {
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
    fn transitions(&self) -> &Transitions {
        &self.transitions
    }

    fn gamma(&self) -> f32 {
        self.gamma
    }
}
