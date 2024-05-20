pub mod common;
pub mod pi;
pub mod vi;

use gymnasium::{common::defs::*, *};
use std::rc::Rc;

/// Markov Decision Process - Sutton & Barto 2018.
pub trait Mdp {
    fn n_s(&self) -> usize;

    fn n_a(&self) -> usize;

    fn transitions(&self) -> Rc<Transitions>;

    fn gamma(&self) -> f32;
}

pub trait MdpSolver<T> {
    fn v_star(&self, s: Discrete) -> f32;

    fn q_star(&self, s: Discrete, a: Discrete) -> Option<f32>;

    fn pi_star(&self, s: Discrete) -> Option<Discrete>;

    #[allow(dead_code)]
    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (T, usize);
}

pub struct MdpSolverPolicy<T> {
    pub mdp_solver: Rc<dyn MdpSolver<T>>,
}

impl<T> Policy<DiscreteSpace, DiscreteSpace> for MdpSolverPolicy<T> {
    fn policy(&self, s: &Discrete) -> Discrete {
        self.mdp_solver.pi_star(*s).unwrap()
    }
}
