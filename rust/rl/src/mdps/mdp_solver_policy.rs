use super::mdp_solver::*;
use gymnasium::{mdps::*, *};
use std::rc::Rc;

pub struct MdpSolverPolicy<T> {
    pub mdp_solver: Rc<dyn MdpSolver<T>>,
}

impl<T> Policy for MdpSolverPolicy<T> {
    fn policy(&self, s: &[ObsActSpaceItem]) -> Vec<ObsActSpaceItem> {
        assert!(s.len() == 1, "Observation state should have just singleton");

        vec![ObsActSpaceItem::Discrete(
            self.mdp_solver
                .pi_star(s[0].discrete_value().unwrap())
                .unwrap(),
        )]
    }
}
