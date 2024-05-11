use super::super::mdp_simulator::*;
use gymnasium::{Continous, Discrete, EpisodeEvent};
use std::iter::zip;
use std::rc::Rc;

/// Ref: https://youtu.be/P0ZvxeQqv0A?si=RLKdOUTNEfKXE63C
//  TODO: get number of states from environment.
pub fn mc_first_visit(
    ep_gen: Rc<dyn EpisodeGenerator>,
    gamma: Continous,
    n_s: usize,
    n_ep: usize,
) -> Vec<Continous> {
    let returns = &mut vec![0 as Continous; n_s];
    let visits = &mut vec![0 as Discrete; n_s];

    let eps = ep_gen.generate(n_ep);
    for e in 0..n_ep {
        let ep = &eps[e];
        let mut g = 0.;
        for t in (0..(ep.len() - 1)).rev() {
            g = gamma * g + ep[t + 1].r;
            if is_first_visit(&ep, t, ep[t].s[0]) {
                returns[ep[t].s[0] as usize] += g;
                visits[ep[t].s[0] as usize] += 1;
            }
        }
    }

    zip(returns, visits)
        .into_iter()
        .map(|(&mut r, &mut v)| if v == 0 { 0. } else { r / (v as Continous) })
        .collect()
}

// TODO: DRY with first visit
/// Ref: https://youtu.be/P0ZvxeQqv0A?si=RLKdOUTNEfKXE63C
pub fn mc_every_visit(
    ep_gen: Rc<dyn EpisodeGenerator>,
    gamma: Continous,
    n_s: usize,
    n_ep: usize,
) -> Vec<Continous> {
    let returns = &mut vec![0 as Continous; n_s];
    let visits = &mut vec![0 as Discrete; n_s];

    let eps = ep_gen.generate(n_ep);
    for e in 0..n_ep {
        let ep = &eps[e];
        let mut g = 0.;
        for t in (0..(ep.len() - 1)).rev() {
            g = gamma * g + ep[t + 1].r;
            returns[ep[t].s[0] as usize] += g;
            visits[ep[t].s[0] as usize] += 1;
        }
    }

    zip(returns, visits)
        .into_iter()
        .map(|(&mut r, &mut v)| if v == 0 { 0. } else { r / (v as Continous) })
        .collect()
}

fn is_first_visit(ep: &[EpisodeEvent], t: usize, s: Discrete) -> bool {
    if t == 0 {
        return true;
    }

    ep.iter().take(t).find(|&x| x.s[0] == s).is_none()
}

#[cfg(test)]
mod tests {
    use super::super::super::super::{
        environments::frozen_lake::*, mdps::mdp::*, mdps::solvers::*,
    };
    use super::*;
    use float_eq::*;

    struct SimpleEnv {
        pub episodes: Vec<Vec<EpisodeEvent>>,
    }

    impl EpisodeGenerator for SimpleEnv {
        fn generate(&self, n: usize) -> Vec<Vec<EpisodeEvent>> {
            self.episodes.clone()
        }
    }

    #[test]
    fn toy_example_with_first_vist() {
        let ep_gen = SimpleEnv {
            episodes: vec![
                vec![
                    EpisodeEvent { s: vec![1], r: -3. },
                    EpisodeEvent { s: vec![4], r: -2. },
                    EpisodeEvent { s: vec![1], r: -1. },
                    EpisodeEvent { s: vec![2], r: -3. },
                    EpisodeEvent { s: vec![1], r: -1. },
                ],
                vec![
                    EpisodeEvent { s: vec![1], r: -3. },
                    EpisodeEvent { s: vec![4], r: -0. },
                ],
                vec![
                    EpisodeEvent { s: vec![2], r: -3. },
                    EpisodeEvent { s: vec![4], r: -0. },
                ],
            ],
        };

        let v = mc_first_visit(Rc::new(ep_gen), 0.9, 6, 3);

        assert_float_eq!(
            v,
            vec![0., (-6.059 / 2.0), (-1. / 2.0), 0., -4.51, 0.],
            abs_all <= 1e-5
        );
    }

    #[test]
    fn toy_example_with_every_vist() {
        let ep_gen = SimpleEnv {
            episodes: vec![
                vec![
                    EpisodeEvent { s: vec![1], r: -3. },
                    EpisodeEvent { s: vec![4], r: -2. },
                    EpisodeEvent { s: vec![1], r: -1. },
                    EpisodeEvent { s: vec![2], r: -3. },
                    EpisodeEvent { s: vec![1], r: -1. },
                ],
                vec![
                    EpisodeEvent { s: vec![1], r: -3. },
                    EpisodeEvent { s: vec![4], r: -0. },
                ],
                vec![
                    EpisodeEvent { s: vec![2], r: -3. },
                    EpisodeEvent { s: vec![4], r: -0. },
                ],
            ],
        };

        let v = mc_every_visit(Rc::new(ep_gen), 0.9, 6, 3);

        assert_float_eq!(
            v,
            vec![
                0.,
                ((-6.059 + -3.0 + -0.9) / 3.0),
                (-1. / 2.0),
                0.,
                -4.51,
                0.
            ],
            abs_all <= 1e-5
        );
    }

    #[test]
    fn mc_first_visit_convergence_large_mdp() {
        let mdp = FrozenLake::new(0.9);
        let ep_gen = EpisodeGeneratorForTransitions {
            transitions: mdp.transitions(),
            seed: 2718,
        };

        let v = mc_first_visit(Rc::new(ep_gen), 0.9, mdp.n_s(), 5000);
        println!("{v:?}");

        assert_float_eq!(
            v,
            vec![0.005391606129871495, 0.00464726549059839, 0.009456750962093424, 0.005755601740507451, 0.008816807029102664, 0.0, 0.027790666037091988, 0.0, 0.025108527690074173, 0.07297545205182163, 0.1380470476684931, 0.0, 0.0, 0.14799431891111725, 0.5034167905442177, 0.0],
            abs_all <= 1e-6
        );
    }
}
