pub mod common;
pub mod mc_methods;
pub mod policy_iteration;
pub mod value_iteration;

use super::super::mdps::mdp_simulator::*;
use gymnasium::{Discrete, EpisodeEvent, Transitions};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::rc::Rc;

struct EpisodeGeneratorForTransitions {
    transitions: Rc<Transitions>,
    seed: u64,
}

impl EpisodeGenerator for EpisodeGeneratorForTransitions {
    fn generate(&self, n: usize) -> Vec<Vec<EpisodeEvent>> {
        let mut eps = vec![];

        let rng = &mut StdRng::seed_from_u64(self.seed);
        for i in 0..n {
            eps.push(vec![]);
            let ep = &mut eps[i];
            let mut s: Discrete = Default::default();
            ep.push(EpisodeEvent {
                s: vec![s],
                r: Default::default(),
            });
            loop {
                let kv = self
                    .transitions
                    .keys()
                    .filter(|&x| x.0 == s)
                    .choose(rng)
                    .unwrap();
                let ts = &self.transitions[kv];
                let dist = WeightedIndex::new(ts.iter().map(|item| item.probability)).unwrap();
                let next = ts.iter().nth(dist.sample(rng)).unwrap();
                ep.push(EpisodeEvent {
                    s: vec![next.next_state],
                    r: next.reward,
                });
                if next.done {
                    break;
                }

                s = next.next_state;
            }
        }

        eps
    }
}
