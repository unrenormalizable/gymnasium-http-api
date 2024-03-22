use gymnasium_rust_client::*;

pub fn discrete_value(space: &ObsActSpace) -> Discrete {
    match space {
        ObsActSpace::Discrete { n } => *n,
        _ => panic!("{space:?} is not ObsActSpace::Discrete."),
    }
}

pub fn discrete_item_value(space: &ObsActSpaceItem) -> Discrete {
    match space {
        ObsActSpaceItem::Discrete(n) => *n,
        _ => panic!("{space:?} is not ObsActSpaceItem::Discrete."),
    }
}
