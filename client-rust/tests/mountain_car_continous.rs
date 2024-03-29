extern crate gymnasium;
extern crate float_eq;
extern crate iced;
extern crate serde_json;
mod common;

use common::*;
use float_eq::*;
use gymnasium::*;
use serde_json::{to_value, Value};
use std::collections::HashMap;

/// Refer: https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/#mountain-car-continuous
#[test]
fn mcc_advanced_make_env_e2e() {
    let c = Client::new("http://localhost", 40004);

    let kwargs = HashMap::<&str, Value>::from([
        ("render_mode", to_value("rgb_array").unwrap()),
        ("goal_velocity", to_value(false).unwrap()),
    ]);
    let env = c.make_env(
        "MountainCarContinuous-v0",
        None,
        Some(false),
        Some(true),
        &kwargs,
    );
    let osvs = box_value(env.observation_space());
    assert_eq!(osvs.0, [2]);
    assert_float_eq!(osvs.1, vec![0.6, 0.07], rmax_all <= 1e-7);
    assert_float_eq!(osvs.2, vec![-1.2, -0.07], rmax_all <= 1e-7);
    let asvs = box_value(env.action_space());
    assert_eq!(asvs.0, [1]);
    assert_float_eq!(asvs.1, vec![1.0], rmax_all <= 1e-7);
    assert_float_eq!(asvs.2, vec![-1.0], rmax_all <= 1e-7);

    let s = env.reset(Some(2718));
    assert_float_eq!(
        continous_items_values(&s),
        vec![-0.546957671, 0.0],
        rmax_all <= 1e-7
    );

    let rf = env.render();
    let data = rf.as_rgb().unwrap();
    assert_eq!((*data.0, *data.1, data.2.len()), (400, 600, 960000));

    let action = env.action_space_sample();
    let si = env.step(&action);
    let obs = continous_items_values(&si.observation);
    assert_eq!(obs.len() as i32, osvs.0[0]);
    assert!(osvs.1[0] >= obs[0] && obs[0] >= osvs.2[0]);
    assert!(osvs.1[1] >= obs[1] && obs[1] >= osvs.2[1]);
    assert!(si.reward < 0.);
    assert!(!si.truncated);
    assert!(!si.terminated);
}
