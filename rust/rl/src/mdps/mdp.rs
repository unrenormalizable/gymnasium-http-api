use gymnasium::*;
use std::rc::Rc;

/// Markov Decision Process - Sutton & Barto 2018.
pub trait Mdp {
    fn n_s(&self) -> usize;

    fn n_a(&self) -> usize;

    fn transitions(&self) -> Rc<Transitions>;

    fn gamma(&self) -> f32;
}
