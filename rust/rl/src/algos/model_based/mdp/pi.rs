use super::common;
use super::*;
use gymnasium::*;
use std::rc::Rc;

// TODO: way to output mini visualizations.

/// https://lcalem.github.io/blog/2018/09/24/sutton-chap04-dp#43-policy-iteration
#[derive(Clone)]
pub struct PolicyIteration {
    n_s: usize,
    n_a: usize,
    transitions: Rc<Transitions>,
    gamma: f32,
    v_init: f32,
    values: Vec<f32>,
    policies: Vec<Discrete>,
}

impl MdpSolver<bool> for PolicyIteration {
    fn v_star(&self, s: Discrete) -> f32 {
        self.values[s as usize]
    }

    fn q_star(&self, s: Discrete, a: Discrete) -> Option<f32> {
        common::q(&self.transitions, self.gamma, &self.values, s, a)
    }

    fn pi_star(&self, s: Discrete) -> Option<Discrete> {
        Some(self.policies[s as usize])
    }

    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (bool, usize) {
        self.values.fill(self.v_init);

        let mut iter_done = 0;
        let mut policy_stable = false;
        for i in 0..num_iterations.unwrap_or(usize::max_value()) {
            iter_done = i;
            self.policy_evaluation(theta);
            policy_stable = self.policy_improvement();
            if num_iterations.is_none() && policy_stable {
                break;
            }
        }

        (policy_stable, iter_done + 1)
    }
}

impl Policy<DiscreteSpace, DiscreteSpace> for PolicyIteration {
    fn policy(&self, s: &Discrete) -> Discrete {
        self.pi_star(*s).unwrap()
    }    
}

#[allow(dead_code)]
impl PolicyIteration {
    pub fn new(mdp: Rc<dyn Mdp>, v_init: f32, a_init: Discrete) -> Self {
        let n_s = mdp.n_s();
        let n_a = mdp.n_a();
        let transitions = mdp.transitions();
        let gamma = mdp.gamma();
        let values = vec![v_init; n_s];
        let policies = vec![a_init; n_s];
        Self {
            n_s,
            n_a,
            transitions,
            gamma,
            v_init,
            values,
            policies,
        }
    }

    fn policy_evaluation(&mut self, theta: f32) {
        loop {
            let delta = self.eval_values_with_curr_policy();
            if theta > delta {
                break;
            }
        }
    }

    fn policy_improvement(&mut self) -> bool {
        let mut policy_stable = true;
        for s in 0..self.n_s {
            let b = self.policies[s];
            self.policies[s] = common::q_for_all_actions(
                &self.transitions,
                self.n_a,
                self.gamma,
                &self.values,
                s as Discrete,
            )
            .0 as Discrete;
            if b != self.policies[s] {
                policy_stable = false;
            }
        }

        policy_stable
    }

    fn eval_values_with_curr_policy(&mut self) -> f32 {
        let mut delta = 0.;

        for s in 0..self.n_s {
            let v_prev = self.values[s];
            self.values[s] = common::q(
                &self.transitions,
                self.gamma,
                &self.values,
                s as Discrete,
                self.policies[s],
            )
            .unwrap_or_default();
            delta = f32::max(delta, f32::abs(v_prev - self.values[s]));
        }

        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::{frozen_lake::*, simple_golf::*};
    use float_eq::assert_float_eq;

    #[test]
    fn first_iteration() {
        let mdp = Rc::new(SimpleGolf::new(0.9));
        let mut p = PolicyIteration::new(mdp, 0., 0);

        let (policy_stable, num_iter) = p.exec(1e-5, Some(1));
        let values = (0..p.n_s)
            .map(|s| p.v_star(s as Discrete))
            .collect::<Vec<_>>();

        assert!(!policy_stable);
        assert_eq!(num_iter, 1);
        // TODO: replace rmax_all with abs
        assert_float_eq!(values, vec![0., 0., 0.], rmax_all <= 1e-3);
    }

    #[test]
    fn without_iterations_policy_is_stabilized() {
        let mdp = Rc::new(SimpleGolf::new(0.9));
        let mut p = PolicyIteration::new(mdp, 0., 0);

        let (policy_stable, num_iter) = p.exec(1e-5, None);
        let values = (0..p.n_s)
            .map(|s| p.v_star(s as Discrete))
            .collect::<Vec<_>>();

        assert!(policy_stable);
        assert_eq!(num_iter, 2);
        assert_float_eq!(values, vec![8.803285, 9.89011, 0.0], rmax_all <= 1e-3);
    }

    #[test]
    fn test_convergence_big_mdp() {
        let mdp = Rc::new(FrozenLake::new(0.9));
        let mut p = PolicyIteration::new(mdp, 0., 0);

        let (policy_stable, iterations) = p.exec(1e-8, Some(10));
        let v_stars = (0..p.n_s)
            .map(|s| p.v_star(s as Discrete))
            .collect::<Vec<_>>();
        let pi_stars = (0..p.n_s)
            .map(|s| p.pi_star(s as Discrete))
            .collect::<Vec<_>>();
        let mut q_stars = vec![];
        for s in 0..p.n_s {
            for a in 0..p.n_a {
                q_stars.push(p.q_star(s as Discrete, a as Discrete));
            }
        }

        assert!(policy_stable);
        assert_eq!(iterations, 10);
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
