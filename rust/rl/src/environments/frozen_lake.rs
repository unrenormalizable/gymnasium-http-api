use crate::Mdp;
use gymnasium_rust_client::*;

/// Gymnasium FrozenLake, default map.
pub struct FrozenLake {
    gamma: f32,
    n_s: usize,
    n_a: usize,
    transitions: Transitions,
}

#[allow(dead_code)]
impl FrozenLake {
    pub fn new(gamma: f32) -> Self {
        let transitions = Transitions::from([
            (
                (7, 2),
                vec![Transition {
                    next_state: 7,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (8, 2),
                vec![
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (3, 2),
                vec![
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (0, 0),
                vec![
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (9, 1),
                vec![
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (3, 3),
                vec![
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (11, 0),
                vec![Transition {
                    next_state: 11,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (13, 2),
                vec![
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (7, 0),
                vec![Transition {
                    next_state: 7,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (7, 3),
                vec![Transition {
                    next_state: 7,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (4, 2),
                vec![
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (1, 2),
                vec![
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (10, 0),
                vec![
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (10, 3),
                vec![
                    Transition {
                        next_state: 11,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (14, 3),
                vec![
                    Transition {
                        next_state: 15,
                        probability: 0.33333334,
                        reward: 1.0,
                        done: true,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (2, 0),
                vec![
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (5, 3),
                vec![Transition {
                    next_state: 5,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (2, 2),
                vec![
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (6, 0),
                vec![
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (11, 3),
                vec![Transition {
                    next_state: 11,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (12, 2),
                vec![Transition {
                    next_state: 12,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (7, 1),
                vec![Transition {
                    next_state: 7,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (8, 1),
                vec![
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (12, 3),
                vec![Transition {
                    next_state: 12,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (14, 2),
                vec![
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 15,
                        probability: 0.33333334,
                        reward: 1.0,
                        done: true,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (0, 2),
                vec![
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (9, 2),
                vec![
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (9, 3),
                vec![
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (15, 2),
                vec![Transition {
                    next_state: 15,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (10, 1),
                vec![
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 11,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (3, 1),
                vec![
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (12, 1),
                vec![Transition {
                    next_state: 12,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (1, 1),
                vec![
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (4, 0),
                vec![
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (12, 0),
                vec![Transition {
                    next_state: 12,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (5, 2),
                vec![Transition {
                    next_state: 5,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (6, 1),
                vec![
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (2, 3),
                vec![
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (13, 1),
                vec![
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (15, 3),
                vec![Transition {
                    next_state: 15,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (1, 3),
                vec![
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (6, 3),
                vec![
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (14, 0),
                vec![
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (15, 1),
                vec![Transition {
                    next_state: 15,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (0, 1),
                vec![
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (6, 2),
                vec![
                    Transition {
                        next_state: 10,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (10, 2),
                vec![
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 11,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (9, 0),
                vec![
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (11, 2),
                vec![Transition {
                    next_state: 11,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (11, 1),
                vec![Transition {
                    next_state: 11,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (2, 1),
                vec![
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 6,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (14, 1),
                vec![
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 15,
                        probability: 0.33333334,
                        reward: 1.0,
                        done: true,
                    },
                ],
            ),
            (
                (5, 0),
                vec![Transition {
                    next_state: 5,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (5, 1),
                vec![Transition {
                    next_state: 5,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (13, 3),
                vec![
                    Transition {
                        next_state: 14,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (1, 0),
                vec![
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (4, 3),
                vec![
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (3, 0),
                vec![
                    Transition {
                        next_state: 3,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 2,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 7,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (0, 3),
                vec![
                    Transition {
                        next_state: 1,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 0,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (13, 0),
                vec![
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                    Transition {
                        next_state: 13,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
            (
                (8, 0),
                vec![
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 12,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (15, 0),
                vec![Transition {
                    next_state: 15,
                    probability: 1.0,
                    reward: 0.0,
                    done: true,
                }],
            ),
            (
                (4, 1),
                vec![
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 5,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: true,
                    },
                ],
            ),
            (
                (8, 3),
                vec![
                    Transition {
                        next_state: 9,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 4,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                    Transition {
                        next_state: 8,
                        probability: 0.33333334,
                        reward: 0.0,
                        done: false,
                    },
                ],
            ),
        ]);

        Self {
            gamma,
            n_s: 16,
            n_a: 4,
            transitions,
        }
    }
}

impl<'a> Mdp<'a> for FrozenLake {
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
