extern crate chrono;
extern crate gymnasium;
extern crate serde_json;

use gymnasium::*;

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
