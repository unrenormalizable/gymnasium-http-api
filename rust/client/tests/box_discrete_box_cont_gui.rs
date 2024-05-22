extern crate float_eq;
extern crate gymnasium;
extern crate iced;
extern crate itertools;
extern crate serde_json;

use float_eq::*;
use gymnasium::{common::defs::*, common::utils::*, *};
use serde_json::to_value;

/// Refer: https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/#mountain-car-continuous
#[test]
fn box_discrete_box_cont_gui() {
    let env = Environment::<BoxSpace<Discrete>, BoxSpace<Continous>>::new(
        "http://127.0.0.1:40004",
        "CarRacing-v2",
        None,
        None,
        None,
        &[("render_mode", to_value("rgb_array").unwrap())],
    );
    assert_eq!(env.name(), "CarRacing-v2");

    let osvs = env.observation_space();
    assert_eq!(osvs.shape, [96, 96, 3]);
    assert!(osvs.low.iter().all(|&x| x == 0));
    assert!(osvs.high.iter().all(|&x| x == 255));
    let asvs = env.action_space();
    assert_eq!(asvs.shape, [3]);
    assert_float_eq!(asvs.low, vec![-1., 0., 0.], rmax_all <= 1e-7);
    assert_float_eq!(asvs.high, vec![1., 1., 1.], rmax_all <= 1e-7);

    let s = env.reset(None);
    assert_eq!(osvs.shape.iter().product::<usize>(), s.len());
    assert_ne!(0, s.iter().filter(|&&x| x != 0).count());

    let rf = env.render();
    let data = rf.as_rgb().unwrap();
    assert_eq!(
        (
            *data.0,
            *data.1,
            deserialize_binary_stream_to_bytes(data.2).len()
        ),
        (400, 600, 960000)
    );
}
