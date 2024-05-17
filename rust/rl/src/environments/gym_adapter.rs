use crate::Mdp;
use gymnasium::*;
use std::rc::Rc;

pub struct GymAdapter {
    env: Rc<Environment<DiscreteSpace, DiscreteSpace>>,
    gamma: f32,
    transitions: Rc<Transitions>,
}

impl GymAdapter {
    pub fn new(env: Rc<Environment<DiscreteSpace, DiscreteSpace>>, gamma: f32) -> Self {
        let transitions = transitions(&env);

        Self {
            env,
            gamma,
            transitions,
        }
    }
}

impl Mdp for GymAdapter {
    fn n_s(&self) -> usize {
        self.env.observation_space().n as usize
    }

    fn n_a(&self) -> usize {
        self.env.action_space().n as usize
    }

    // TODO: Should the transitions out of end states be removed.
    fn transitions(&self) -> Rc<Transitions> {
        Rc::clone(&self.transitions)
    }

    fn gamma(&self) -> f32 {
        self.gamma
    }
}
