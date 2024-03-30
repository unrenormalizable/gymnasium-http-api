use gymnasium::*;
use std::option::*;

pub trait MdpSolver<T> {
    fn v_star(&self, s: Discrete) -> f32;

    fn q_star(&self, s: Discrete, a: Discrete) -> Option<f32>;

    fn pi_star(&self, s: Discrete) -> Option<Discrete>;

    fn exec(&mut self, theta: f32, num_iterations: Option<usize>) -> (T, usize);
}
