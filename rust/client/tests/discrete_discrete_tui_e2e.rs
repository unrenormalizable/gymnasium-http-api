extern crate float_eq;
extern crate gymnasium;
extern crate iced;
extern crate serde_json;

use float_eq::*;
use gymnasium::*;
use serde_json::to_value;

#[test]
fn discrete_discrete_tui_e2e() {
    let env = Environment::<DiscreteSpace, DiscreteSpace>::new(
        "http://127.0.0.1:40004",
        "FrozenLake-v1",
        Some(1),
        Some(false),
        Some(true),
        &[
            ("render_mode", to_value("ansi").unwrap()),
            ("is_slippery", to_value(false).unwrap()),
            ("desc", to_value(["GGGH", "GSGH", "GGGF", "FFFG"]).unwrap()),
        ],
    );
    assert_eq!(env.name(), "FrozenLake-v1");
    assert_eq!(env.observation_space().n, 16);
    assert_eq!(env.action_space().n, 4);
    assert_eq!(transitions(&env).len(), 64);

    let s = env.reset(Some(2718));
    assert_eq!(s, 5);

    let rf = env.render();
    assert_eq!(
        rf.as_str().unwrap(),
        "\nGGGH\nG\u{1b}[41mS\u{1b}[0mGH\nGGGF\nFFFG\n"
    );

    let si = env.step(&1);
    assert_eq!(si.observation, 9);
    assert_eq!(format!("terminated: {}", si.terminated), "terminated: true");
    assert_eq!(format!("truncated: {}", si.truncated), "truncated: true");
    assert_float_eq!(si.reward, 1., rmax <= 1e-16);

    let rf = env.render();
    assert_eq!(
        rf.as_str().unwrap(),
        "  (Down)\nGGGH\nGSGH\nG\u{1b}[41mG\u{1b}[0mGF\nFFFG\n"
    );
}
