use super::mdp_solver::*;
use gymnasium::{*, defs::policy::*};
use std::rc::Rc;

pub struct MdpSolverPolicy<T> {
    pub mdp_solver: Rc<dyn MdpSolver<T>>,
}

impl<T> Policy<DiscreteSpace, DiscreteSpace> for MdpSolverPolicy<T> {
    fn policy(&self, s: &Discrete) -> Discrete {
        self.mdp_solver.pi_star(*s).unwrap()
    }
}
