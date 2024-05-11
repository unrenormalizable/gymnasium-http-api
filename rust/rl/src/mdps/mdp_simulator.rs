use gymnasium::{Discrete, EpisodeEvent};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub trait EpisodeGenerator {
    fn generate(&self, n: usize) -> Vec<Vec<EpisodeEvent>>;
}

pub trait MdpSimulator {
    fn name(&self) -> String;

    fn n_s() -> Discrete;

    fn action_space_sample(&self);

    fn reset(&self);

    fn step(&self);
}

pub trait Weighted<S> {
    fn s(&self) -> S;

    fn p(&self) -> f32;
}

pub fn pick_next<T, S>(rng: &mut StdRng, ts: &Vec<T>) -> S
where
    T: Weighted<S>,
{
    let dist = WeightedIndex::new(ts.iter().map(|item| item.p())).unwrap();
    ts.iter().nth(dist.sample(rng)).unwrap().s()
}

#[cfg(test)]
mod tests {
    use super::{pick_next, Weighted};
    use float_eq::*;
    use rand::prelude::*;

    #[test]
    fn test_pick_next_unseeded() {
        let items = &mut vec![
            TX {
                s: 0,
                p: 0.2,
                count: 0,
            },
            TX {
                s: 1,
                p: 0.8,
                count: 0,
            },
        ];

        let rng = &mut StdRng::from_entropy();
        let n = 10000;
        for _ in 0..n {
            let i = pick_next(rng, items);
            items[i].count += 1;
        }

        assert_float_eq!(items[0].count as f32 / n as f32, 0.2, abs <= 1e-2);
        assert_float_eq!(items[1].count as f32 / n as f32, 0.8, abs <= 1e-2);
    }

    struct TX {
        pub s: usize,
        pub p: f32,
        pub count: i32,
    }

    impl Weighted<usize> for TX {
        fn p(&self) -> f32 {
            self.p
        }

        fn s(&self) -> usize {
            self.s
        }
    }
}
