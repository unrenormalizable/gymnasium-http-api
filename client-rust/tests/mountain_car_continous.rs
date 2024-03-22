extern crate float_eq;
extern crate gymnasium_rust_client;
extern crate serde_json;
mod common;

use common::*;
use float_eq::*;
use gymnasium_rust_client::*;
use serde_json::{to_value, Value};
use std::collections::HashMap;

/// Refer: https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/#mountain-car-continuous
#[test]
fn mcc_advanced_make_env_e2e() {
    let c = GymClient::new("http://localhost", 40004);

    let kwargs = HashMap::<&str, Value>::from([
        ("render_mode", to_value("rgb_array").unwrap()),
        ("goal_velocity", to_value(false).unwrap()),
    ]);
    let env = c.make_env(
        "MountainCarContinuous-v0",
        Some(1),
        Some(false),
        Some(true),
        kwargs,
    );
    let bv = box_value(env.observation_space());
    assert_eq!(bv.0, [2]);
    assert_float_eq!(bv.1, vec![0.6, 0.07], rmax_all <= 1e-7);
    assert_float_eq!(bv.2, vec![-1.2, -0.07], rmax_all <= 1e-7);
    let bv = box_value(env.action_space());
    assert_eq!(bv.0, [1]);
    assert_float_eq!(bv.1, vec![1.0], rmax_all <= 1e-7);
    assert_float_eq!(bv.2, vec![-1.0], rmax_all <= 1e-7);

    let s = env.reset(Some(2718));
    assert_float_eq!(
        continous_items_values(&s),
        vec![-0.546957671, 0.0],
        rmax_all <= 1e-7
    );

    let rf = env.render();
    let rgb = rf.as_rgb().unwrap();
    assert_eq!((rgb.len(), rgb[0].len(), rgb[0][0].len()), (400, 600, 3));

    //let si = env.step(&env.action_space_sample());
    //assert_eq!(discrete_item_value(&si.observation[0]), 9);
    //assert_eq!(format!("terminated: {}", si.terminated), "terminated: true");
    //assert_eq!(format!("truncated: {}", si.truncated), "truncated: true");
    //assert_float_eq!(si.reward, 1., rmax <= 1e-16);

    //let rf = env.render();
    //assert_eq!(rf, "  (Down)\nGGGH\nGSGH\nG\u{1b}[41mG\u{1b}[0mGF\nFFFG\n");

    //Ok(())
}
