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
            size: iced::Size {
                height: 500.,
                width: 650.,
            },
            // TODO: icon.
            ..window::Settings::default()
        },
        ..Settings::with_flags(ui::display::EnvironmentProxyFlags {
            api_url: "http://127.0.0.1:40004",
            env_name: "MountainCar-v0",
            ..ui::display::EnvironmentProxyFlags::default()
        })
    })
}
