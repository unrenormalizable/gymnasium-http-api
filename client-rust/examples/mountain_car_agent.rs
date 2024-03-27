extern crate chrono;
extern crate gymnasium;
extern crate serde_json;

use base64::prelude::*;
use gymnasium::*;
use image::RgbImage;
use ndarray::Array3;
use serde_json::{to_value, Value};
use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;

use iced::window;
use iced::{Application, Settings};

fn main() -> ui::Result {
    tracing_subscriber::fmt::init();

    ui::GymnasiumApp::run(Settings {
        antialiasing: true,
        window: window::Settings {
            position: window::Position::Centered,
            ..window::Settings::default()
        },
        ..Settings::default()
    })
}

#[allow(dead_code)]
fn main2() {
    let (tx, rx) = mpsc::channel::<(chrono::NaiveTime, usize, RenderFrame)>();

    _ = thread::spawn(move || {
        for (ts, n, rf) in rx {
            let payload = rf.as_rgb().unwrap();
            let bytes = BASE64_STANDARD.decode(payload.2).unwrap();
            let arr = Array3::from_shape_vec((*payload.0, *payload.1, 3), bytes).unwrap();
            let img = array_to_image(arr);
            img.save(format!("D:\\src\\delme\\img\\img{n}.bmp"))
                .unwrap();
            let ms = (chrono::Utc::now().time() - ts).num_milliseconds() as f32;
            println!(
                "#{}: {}ms -> {}x{}. Frame rate: {}.",
                n,
                ms,
                payload.0,
                payload.1,
                1000.0 / ms
            );
        }
    });

    let c = Client::new("http://127.0.0.1", 40004);
    let kwargs = HashMap::<&str, Value>::from([("render_mode", to_value("rgb_array").unwrap())]);
    let env = c.make_env("MountainCar-v0", None, None, None, &kwargs);

    for ep in 0..1000 {
        env.reset(Some(2718));
        let mut tot_reward = 0.;
        for i in 0..100 {
            let action = env.action_space_sample();
            let state = env.step(&action);
            let ts = chrono::Utc::now().time();
            let render_frame = env.render();
            tx.send((ts, i, render_frame)).unwrap();
            tot_reward += state.reward;

            if state.truncated || state.terminated {
                break;
            }
        }
        println!("Finished episode {} with total reward {}", ep, tot_reward);
    }
}

fn array_to_image(arr: Array3<u8>) -> RgbImage {
    assert!(arr.is_standard_layout());

    let (height, width, _) = arr.dim();
    let raw = arr.into_raw_vec();

    RgbImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}
