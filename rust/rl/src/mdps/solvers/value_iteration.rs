use super::common;
use crate::mdps::mdp::*;
use crate::mdps::mdp_solver::*;
use gymnasium_rust_client::*;

/// https://lcalem.github.io/blog/2018/09/24/sutton-chap04-dp#44-value-iteration
pub struct ValueIteration<'a> {
    n_s: usize,
    n_a: usize,
    transitions: &'a Transitions,
    gamma: f32,
    v_init: f32,
    values: Vec<f32>,
    values_prev: Vec<f32>,
}

impl<'a> MdpSolver<f32> for ValueIteration<'a> {
    fn v_star(&self, s: Discrete) -> f32 {
        self.values[s as usize]
    }

    fn q_star(&self, s: Discrete, a: Discrete) -> Option<f32> {
        common::q(self.transitions, self.gamma, &self.values, s, a)
    }

    fn pi_star(&self, s: Discrete) -> Option<Discrete> {
        let max =
            common::q_for_all_actions(self.transitions, self.n_a, self.gamma, &self.values, s);
        max.1.map_or_else(|| None, |_| Some(max.0 as Discrete))
    }

    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (f32, usize) {
        self.values.fill(self.v_init);
        self.values_prev.fill(self.v_init);
        let mut delta = 0.;
        let mut iter_done = 0;
        for i in 0..num_iterations.unwrap_or(usize::max_value()) {
            iter_done = i;
            self.values_prev.copy_from_slice(&self.values);
            self.values.fill(self.v_init);
            delta = self.one_value_iteration();
            if num_iterations.is_none() && theta > delta {
                break;
            }
        }

        self.values_prev.fill(self.v_init);
        (delta, iter_done + 1)
    }
}

#[allow(dead_code)]
impl<'a> ValueIteration<'a> {
    pub fn new(mdp: &'a dyn Mdp<'a>, v_init: f32) -> Self {
        let n_s = mdp.n_s();
        let n_a = mdp.n_a();
        let transitions = mdp.transitions();
        let gamma = mdp.gamma();
        let values = vec![v_init; n_s];
        let values_prev = vec![v_init; n_s];
        Self {
            n_s,
            n_a,
            transitions,
            v_init,
            gamma,
            values,
            values_prev,
        }
    }

    fn one_value_iteration(&mut self) -> f32 {
        let mut delta = 0.;

        for s in 0..self.n_s {
            let v_new = common::q_for_all_actions(
                self.transitions,
                self.n_a,
                self.gamma,
                &self.values_prev,
                s as Discrete,
            )
            .1
            .unwrap_or_default();
            delta = f32::max(delta, f32::abs(self.values_prev[s] - v_new));
            self.values[s] = v_new;
        }

        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::{frozen_lake::*, simple_golf::*};
    use float_eq::assert_float_eq;

    #[test]
    fn first_iteration() {
        let mdp = SimpleGolf::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0.);

        let (_, num_iter) = vi.exec(1e-5, Some(1));
        let values = (0..vi.n_s)
            .map(|s| vi.v_star(s as Discrete))
            .collect::<Vec<_>>();

        assert_eq!(num_iter, 1);
        assert_float_eq!(values, vec![0., 9., 0.], rmax_all <= 0.001);
    }

    #[test]
    fn nth_iteration() {
        let mdp = SimpleGolf::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0.);

        let (delta, iterations) = vi.exec(1e-4, Some(6));
        let values = (0..vi.n_s)
            .map(|s| vi.v_star(s as Discrete))
            .collect::<Vec<_>>();

        assert_eq!(iterations, 6);
        assert_float_eq!(delta, 0.00239, rmax <= 0.001);
        assert_float_eq!(values, vec![8.80299, 9.89010, 0.], rmax_all <= 0.0001);
    }

    #[test]
    fn test_policy() {
        let mdp = SimpleGolf::new(0.9);
        let mut vi = ValueIteration::new(&mdp, 0.);

        vi.exec(1e-4, Some(6));

        assert_eq!(vi.pi_star(0), Some(0));
        assert_eq!(vi.pi_star(1), Some(2));
        assert_eq!(vi.pi_star(2), None);
    }

    #[test]
    fn test_convergence_big_mdp() {
        let mdp = FrozenLake::new(0.9);
        let mut v = ValueIteration::new(&mdp, 0.);

        let (delta, iterations) = v.exec(1e-8, Some(100));
        let v_stars = (0..v.n_s)
            .map(|s| v.v_star(s as Discrete))
            .collect::<Vec<_>>();
        let pi_stars = (0..v.n_s)
            .map(|s| v.pi_star(s as Discrete))
            .collect::<Vec<_>>();
        let mut q_stars = vec![];
        for s in 0..v.n_s {
            for a in 0..v.n_a {
                q_stars.push(v.q_star(s as Discrete, a as Discrete));
            }
        }

        assert_eq!(iterations, 100);
        assert_float_eq!(delta, 4.4703484e-8, rmax <= 1e-16);
        assert_float_eq!(
            v_stars,
            vec![
                0.06889083,
                0.061414514,
                0.07440972,
                0.055807278,
                0.09185447,
                0.0,
                0.11220818,
                0.0,
                0.14543629,
                0.2474969,
                0.29961756,
                0.0,
                0.0,
                0.37993586,
                0.6390201,
                0.0
            ],
            rmax_all <= 1e-5
        );
        assert_eq!(
            pi_stars,
            vec![
                Some(0),
                Some(3),
                Some(0),
                Some(3),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(3),
                Some(1),
                Some(0),
                Some(0),
                Some(0),
                Some(2),
                Some(1),
                Some(0)
            ]
        );
        //assert!((0..v.n_a).map(|a| v.q_star(2, a)).all(|q| q.is_none()));
        assert_float_eq!(
            q_stars
                .iter()
                .map(|q| q.unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![
                0.06889084,
                0.06664795,
                0.06664795,
                0.05975885,
                0.039091602,
                0.042990163,
                0.04074727,
                0.061414517,
                0.07440972,
                0.06882899,
                0.07272755,
                0.057489455,
                0.0390651,
                0.0390651,
                0.033484366,
                0.05580728,
                0.091854475,
                0.07118723,
                0.06429813,
                0.048223592,
                0.0,
                0.0,
                0.0,
                0.0,
                0.11220818,
                0.089885265,
                0.11220818,
                0.022322915,
                0.0,
                0.0,
                0.0,
                0.0,
                0.07118723,
                0.11787996,
                0.10180542,
                0.1454363,
                0.15761164,
                0.2474969,
                0.20386602,
                0.13351615,
                0.29961756,
                0.2659551,
                0.22536848,
                0.10791153,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.18822983,
                0.30568677,
                0.37993586,
                0.2659551,
                0.39557207,
                0.6390201,
                0.61492467,
                0.5371994,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            rmax_all <= 1e-5
        );
    }
}
