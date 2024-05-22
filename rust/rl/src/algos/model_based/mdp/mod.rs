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

pub trait MdpSolver<T> : Policy<DiscreteSpace, DiscreteSpace> {
    fn v_star(&self, s: Discrete) -> f32;

    fn q_star(&self, s: Discrete, a: Discrete) -> Option<f32>;

    fn pi_star(&self, s: Discrete) -> Option<Discrete>;

    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (T, usize);
}
