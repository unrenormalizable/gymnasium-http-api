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

#[test]
fn fl_advanced_make_env_e2e() {
    let c = Client::new("http://localhost", 40004);

    let kwargs = HashMap::<&str, Value>::from([
        ("render_mode", to_value("ansi").unwrap()),
        ("is_slippery", to_value(false).unwrap()),
        ("desc", to_value(["GGGH", "GSGH", "GGGF", "FFFG"]).unwrap()),
    ]);
    let env = c.make_env("FrozenLake-v1", Some(1), Some(false), Some(true), &kwargs);
    assert_eq!(discrete_value(env.observation_space()), 16);
    assert_eq!(discrete_value(env.action_space()), 4);
    assert_ne!(env.transitions().len(), 0);

    let s = env.reset(Some(2718));
    assert_eq!(discrete_item_value(&s[0]), 5);

    let rf = env.render();
    assert_eq!(
        rf.as_str().unwrap(),
        "\nGGGH\nG\u{1b}[41mS\u{1b}[0mGH\nGGGF\nFFFG\n"
    );

    let si = env.step(&[ObsActSpaceItem::Discrete(1)]);
    assert_eq!(discrete_item_value(&si.observation[0]), 9);
    assert_eq!(format!("terminated: {}", si.terminated), "terminated: true");
    assert_eq!(format!("truncated: {}", si.truncated), "truncated: true");
    assert_float_eq!(si.reward, 1., rmax <= 1e-16);

    let rf = env.render();
    assert_eq!(
        rf.as_str().unwrap(),
        "  (Down)\nGGGH\nGSGH\nG\u{1b}[41mG\u{1b}[0mGF\nFFFG\n"
    );
}
