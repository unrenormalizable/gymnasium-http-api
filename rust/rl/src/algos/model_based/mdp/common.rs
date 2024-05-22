use crate::math;
use gymnasium::common::defs::*;

pub fn q_for_all_actions(
    transitions: &Transitions,
    n_a: usize,
    gamma: f32,
    v: &[f32],
    s: Discrete,
) -> (usize, Option<f32>) {
    let qs: Vec<_> = (0..n_a)
        .map(|a| q(transitions, gamma, v, s, a as Discrete))
        .collect();

    math::max(&qs)
}

pub fn q(
    transitions: &Transitions,
    gamma: f32,
    v: &[f32],
    s: Discrete,
    a: Discrete,
) -> Option<f32> {
    if let Some(ts) = transitions.get(&(s, a)) {
        let q = ts
            .iter()
            .map(|t| t.probability as f32 * (t.reward as f32 + gamma * v[t.next_state as usize]))
            .sum();
        Some(q)
    } else {
        None
    }
}
