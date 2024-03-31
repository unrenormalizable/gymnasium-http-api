use super::*;

pub trait Policy {
    fn policy(&self, s: &[ObsActSpaceItem]) -> Vec<ObsActSpaceItem>;
}

pub struct RandomEnvironmentPolicy {
    pub env: Rc<Environment>,
}

impl Policy for RandomEnvironmentPolicy {
    fn policy(&self, _s: &[ObsActSpaceItem]) -> Vec<ObsActSpaceItem> {
        self.env.action_space_sample()
    }
}
