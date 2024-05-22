use crate::*;

pub trait Policy<O: Space, A: Space> {
    fn policy(&self, s: &O::Item) -> A::Item;
}
