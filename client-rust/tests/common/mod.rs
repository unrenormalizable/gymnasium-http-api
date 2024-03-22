use gymnasium_rust_client::*;

#[allow(dead_code)]
pub fn discrete_value(space: &ObsActSpace) -> Discrete {
    match space {
        ObsActSpace::Discrete { n } => *n,
        _ => panic!("{space:?} is not ObsActSpace::Discrete."),
    }
}

#[allow(dead_code)]
pub fn box_value(space: &ObsActSpace) -> (Vec<Discrete>, Vec<Continous>, Vec<Continous>) {
    match space {
        ObsActSpace::Box { shape, high, low } => (shape.clone(), high.clone(), low.clone()),
        _ => panic!("{space:?} is not ObsActSpace::Box."),
    }
}

#[allow(dead_code)]
pub fn discrete_item_value(item: &ObsActSpaceItem) -> Discrete {
    match item {
        ObsActSpaceItem::Discrete(n) => *n,
        _ => panic!("{item:?} is not ObsActSpaceItem::Discrete."),
    }
}

#[allow(dead_code)]
pub fn continous_item_value(item: &ObsActSpaceItem) -> Continous {
    match item {
        ObsActSpaceItem::Box(n) => *n,
        _ => panic!("{item:?} is not ObsActSpaceItem::Discrete."),
    }
}

#[allow(dead_code)]
pub fn continous_items_values(items: &[ObsActSpaceItem]) -> Vec<Continous> {
    items.iter().map(continous_item_value).collect()
}
