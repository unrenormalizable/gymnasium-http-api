use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

pub type Discrete = i64;
pub type Continous = f64;

pub trait Policy<O: Space, A: Space> {
    fn policy(&self, s: &O::Item) -> A::Item;
}

#[derive(Debug)]
pub struct Transition {
    pub next_state: Discrete,
    pub probability: Continous,
    pub reward: f64,
    pub done: bool,
}

pub type Transitions = HashMap<(Discrete, Discrete), Vec<Transition>>;

pub trait Space {
    type Item;

    fn new(val: &Value) -> Self;

    fn action(val: &Value) -> Self::Item;

    fn observation(val: &Value) -> Self::Item;

    fn action_request(actions: &Self::Item) -> HashMap<&str, Value>;
}

#[derive(Clone, Debug, Deserialize)]
pub struct EpisodeEvent<O> {
    pub s: Vec<O>,
    pub r: Continous,
}

pub trait EpisodeGenerator<O>
where
    for<'de> O: Deserialize<'de>,
{
    fn generate(&self, n: usize, seed: Option<usize>) -> Vec<Vec<EpisodeEvent<O>>>;
}
