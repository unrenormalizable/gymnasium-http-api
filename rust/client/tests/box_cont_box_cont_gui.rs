extern crate float_eq;
extern crate gymnasium;
extern crate iced;
extern crate serde_json;

use float_eq::*;
use gymnasium::{common::utils::*, *};
use serde_json::to_value;

/// Refer: https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/#mountain-car-continuous
#[test]
fn box_cont_box_cont_gui() {
    let env = Environment::<BoxSpace<Continous>, BoxSpace<Continous>>::new(
        "http://127.0.0.1:40004",
        "MountainCarContinuous-v0",
        None,
        Some(false),
        Some(true),
        &[
            ("render_mode", to_value("rgb_array").unwrap()),
            ("goal_velocity", to_value(false).unwrap()),
        ],
    );
    assert_eq!(env.name(), "MountainCarContinuous-v0");

    let osvs = env.observation_space();
    assert_eq!(osvs.shape, [2]);
    assert_float_eq!(osvs.high, vec![0.6, 0.07], rmax_all <= 1e-7);
    assert_float_eq!(osvs.low, vec![-1.2, -0.07], rmax_all <= 1e-7);
    let asvs = env.action_space();
    assert_eq!(asvs.shape, [1]);
    assert_float_eq!(asvs.high, vec![1.0], rmax_all <= 1e-7);
    assert_float_eq!(asvs.low, vec![-1.0], rmax_all <= 1e-7);

    let s = env.reset(Some(2718));
    assert_float_eq!(s, vec![-0.546957671, 0.0], rmax_all <= 1e-7);

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

    let action = env.action_space_sample();
    let si = env.step(&action);
    let obs = si.observation;
    assert_eq!(obs.len(), osvs.shape[0] as usize);
    assert!(osvs.high[0] >= obs[0] && obs[0] >= osvs.low[0]);
    assert!(osvs.high[1] >= obs[1] && obs[1] >= osvs.low[1]);
    assert!(si.reward < 0.);
    assert!(!si.truncated);
    assert!(!si.terminated);
}
