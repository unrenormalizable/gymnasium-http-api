use crate::Mdp;
use gymnasium::*;

/// https://towardsdatascience.com/reinforcement-learning-an-easy-introduction-to-value-iteration-e4cfe0731fd5
pub struct SimpleGolf {
    gamma: f32,
    n_s: usize,
    n_a: usize,
    transitions: Transitions,
}

#[allow(dead_code)]
impl SimpleGolf {
    pub fn new(gamma: f32) -> Self {
        let transitions = Transitions::from([
            (
                (0, 0),
                vec![
                    Transition {
                        next_state: 1,
                        probability: 0.9,
                        reward: 0.,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.1,
                        reward: 0.,
                        done: false,
                    },
                ],
            ),
            (
                (1, 1),
                vec![
                    Transition {
                        next_state: 0,
                        probability: 0.9,
                        reward: 0.,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.1,
                        reward: 0.,
                        done: false,
                    },
                ],
            ),
            (
                (1, 2),
                vec![
                    Transition {
                        next_state: 2,
                        probability: 0.9,
                        reward: 10.,
                        done: true,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.1,
                        reward: 0.,
                        done: false,
                    },
                ],
            ),
        ]);

        Self {
            gamma,
            n_s: 3,
            n_a: 3,
            transitions,
        }
    }
}

impl<'a> Mdp<'a> for SimpleGolf {
    fn n_s(&self) -> usize {
        self.n_s
    }

    fn n_a(&self) -> usize {
        self.n_a
    }

    fn transitions(&'a self) -> &'a Transitions {
        &self.transitions
    }

    fn gamma(&self) -> f32 {
        self.gamma
    }
}
