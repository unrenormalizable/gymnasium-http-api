use super::*;

pub trait Policy<O: Space, A: Space> {
    fn policy(&self, s: &O::Item) -> A::Item;
}

pub struct RandomEnvironmentPolicy<O: Space, A: Space> {
    pub env: Rc<Environment<O, A>>,
}

impl<O: Space, A: Space> Policy<O, A> for RandomEnvironmentPolicy<O, A> {
    fn policy(&self, _s: &O::Item) -> A::Item {
        self.env.action_space_sample()
    }
}
