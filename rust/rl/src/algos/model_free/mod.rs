pub mod gradient_free;

use gymnasium::common::defs::*;

pub trait MdpSimulator {
    fn name(&self) -> String;

    fn n_s() -> Discrete;

    fn action_space_sample(&self);

    fn reset(&self);

    fn step(&self);
}
