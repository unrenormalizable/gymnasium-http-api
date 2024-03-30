use gymnasium::*;

/// Markov Decision Process - Sutton & Barto 2018.
pub trait Mdp<'a> {
    fn n_s(&self) -> usize;

    fn n_a(&self) -> usize;

    fn transitions(&'a self) -> &'a Transitions;

    fn gamma(&self) -> f32;
}
