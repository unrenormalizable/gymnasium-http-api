use gymnasium::common::defs::{Continous, Discrete, EpisodeEvent, EpisodeGenerator};
use std::iter::zip;
use std::rc::Rc;

/// Ref: https://youtu.be/P0ZvxeQqv0A?si=RLKdOUTNEfKXE63C
#[allow(dead_code)]
pub fn mc_first_visit(
    ep_gen: Rc<dyn EpisodeGenerator<Discrete>>,
    gamma: Continous,
    n_s: usize,
    n_ep: usize,
) -> Vec<Continous> {
    mc_first_core(ep_gen, gamma, n_s, n_ep, is_first_visit)
}

// TODO: DRY with first visit
/// Ref: https://youtu.be/P0ZvxeQqv0A?si=RLKdOUTNEfKXE63C
#[allow(dead_code)]
pub fn mc_every_visit(
    ep_gen: Rc<dyn EpisodeGenerator<Discrete>>,
    gamma: Continous,
    n_s: usize,
    n_ep: usize,
) -> Vec<Continous> {
    mc_first_core(ep_gen, gamma, n_s, n_ep, |_, _, _| true)
}

#[allow(dead_code)]
pub fn mc_first_core(
    ep_gen: Rc<dyn EpisodeGenerator<Discrete>>,
    gamma: Continous,
    n_s: usize,
    n_ep: usize,
    is_first_visit: fn(&[EpisodeEvent<Discrete>], usize, Discrete) -> bool,
) -> Vec<Continous> {
    let returns = &mut vec![0 as Continous; n_s];
    let visits = &mut vec![0 as Discrete; n_s];

    let eps = ep_gen.generate(n_ep, None);
    for ep in eps.iter().take(n_ep) {
        let mut g = 0.;
        for t in (0..(ep.len() - 1)).rev() {
            g = gamma * g + ep[t + 1].r;
            if is_first_visit(ep, t, ep[t].s[0]) {
                returns[ep[t].s[0] as usize] += g;
                visits[ep[t].s[0] as usize] += 1;
            }
        }
    }

    zip(returns, visits)
        .map(|(&mut r, &mut v)| if v == 0 { 0. } else { r / (v as Continous) })
        .collect()
}

fn is_first_visit(ep: &[EpisodeEvent<Discrete>], t: usize, s: Discrete) -> bool {
    if t == 0 {
        return true;
    }

    !ep.iter().take(t).any(|x| x.s[0] == s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::*;

    struct SimpleEnv {
        pub episodes: Vec<Vec<EpisodeEvent<Discrete>>>,
    }

    impl EpisodeGenerator<Discrete> for SimpleEnv {
        fn generate(&self, _n: usize, _seed: Option<usize>) -> Vec<Vec<EpisodeEvent<Discrete>>> {
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
}
