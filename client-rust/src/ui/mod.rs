use display::Display;

use iced::executor;
use iced::theme::{self, Theme};
use iced::time;
use iced::widget::{button, column, container, row, slider, text};
use iced::{Alignment, Application, Command, Element, Length, Subscription};
use std::time::Duration;

pub type Result = iced::Result;

pub struct GymnasiumApp {
    grid: Display,
    is_playing: bool,
    queued_ticks: usize,
    speed: usize,
    next_speed: Option<usize>,
    version: usize,
}

#[derive(Debug, Clone)]
pub enum Message {
    Display(display::Message, usize),
    Tick,
    TogglePlayback,
    Next,
    SpeedChanged(f32),
}

impl Application for GymnasiumApp {
    type Message = Message;
    type Theme = Theme;
    type Executor = executor::Default;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        (
            Self {
                grid: Display::new(),
                is_playing: Default::default(),
                queued_ticks: Default::default(),
                speed: 30,
                next_speed: Default::default(),
                version: Default::default(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Gymnasium - <TBD: Environment title title>")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Display(message, version) => {
                if version == self.version {
                    self.grid.update(message);
                }
            }
            Message::Tick | Message::Next => {
                self.queued_ticks = (self.queued_ticks + 1).min(self.speed);

                if let Some(task) = self.grid.tick(self.queued_ticks) {
                    if let Some(speed) = self.next_speed.take() {
                        self.speed = speed;
                    }

                    self.queued_ticks = 0;

                    let version = self.version;

                    return Command::perform(task, move |message| {
                        Message::Display(message, version)
                    });
                }
            }
            Message::TogglePlayback => {
                self.is_playing = !self.is_playing;
            }
            Message::SpeedChanged(speed) => {
                if self.is_playing {
                    self.next_speed = Some(speed.round() as usize);
                } else {
                    self.speed = speed.round() as usize;
                }
            }
        }

        Command::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.is_playing {
            time::every(Duration::from_millis(1000 / self.speed as u64)).map(|_| Message::Tick)
        } else {
            Subscription::none()
        }
    }

    fn view(&self) -> Element<Message> {
        let version = self.version;
        let selected_speed = self.next_speed.unwrap_or(self.speed);
        let controls = view_controls(self.is_playing, selected_speed);

        let content = column![
            self.grid
                .view()
                .map(move |message| Message::Display(message, version)),
            controls,
        ]
        .height(Length::Fill);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}

fn view_controls<'a>(is_playing: bool, speed: usize) -> Element<'a, Message> {
    let playback_controls = row![
        button(if is_playing { "Pause" } else { "Play" }).on_press(Message::TogglePlayback),
        button("Next")
            .on_press(Message::Next)
            .style(theme::Button::Secondary),
    ]
    .spacing(10);

    let speed_controls = row![
        slider(1.0..=1000.0, speed as f32, Message::SpeedChanged),
        text(format!("x{speed}")).size(16),
    ]
    .align_items(Alignment::Center)
    .spacing(10);

    row![playback_controls, speed_controls,]
        .padding(10)
        .spacing(20)
        .align_items(Alignment::Center)
        .into()
}

mod display {
    use iced::{Element, Length};
    use std::future::Future;
    use std::time::{Duration, Instant};

    pub struct Display {
        state: State,
        last_tick_duration: Duration,
        last_queued_ticks: usize,
    }

    #[derive(Debug, Clone)]
    pub enum Message {
        Ticked {
            result: Result<(), TickError>,
            tick_duration: Duration,
        },
    }

    #[derive(Debug, Clone)]
    pub enum TickError {
        JoinFailed,
    }

    impl Display {
        pub fn new() -> Self {
            Self {
                state: State::with_env(EnvironmentProxy::new()),
                last_tick_duration: Duration::default(),
                last_queued_ticks: 0,
            }
        }
    }

    impl Display {
        pub fn tick(&mut self, amount: usize) -> Option<impl Future<Output = Message>> {
            let tick = self.state.tick(amount)?;

            self.last_queued_ticks = amount;

            Some(async move {
                let start = Instant::now();
                let result = tick.await;
                let tick_duration = start.elapsed() / amount as u32;

                Message::Ticked {
                    result,
                    tick_duration,
                }
            })
        }

        pub fn update(&mut self, message: Message) {
            match message {
                Message::Ticked {
                    result: Ok(()),
                    tick_duration,
                } => {
                    self.state.update();

                    self.last_tick_duration = tick_duration;
                }
                Message::Ticked {
                    result: Err(error), ..
                } => {
                    dbg!(error);
                }
            }
        }

        pub fn view(&self) -> Element<Message> {
            use base64::prelude::*;

            let rgb = self.state.get_rf();
            let rgb = rgb.as_rgb().unwrap();
            let bytes = base64::prelude::BASE64_STANDARD.decode(rgb.2).unwrap();

            let handle =
                iced::widget::image::Handle::from_pixels(*rgb.1 as u32, *rgb.0 as u32, bytes);
            let image = iced::widget::Image::new(handle)
                .width(Length::Fill)
                .height(Length::Fill);

            iced::widget::container(image)
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x()
                .center_y()
                .into()
        }
    }

    struct State {
        env: EnvironmentProxy,
        is_ticking: bool,
    }

    impl State {
        pub fn with_env(env: EnvironmentProxy) -> Self {
            Self {
                env,
                is_ticking: Default::default(),
            }
        }

        fn get_rf(&self) -> crate::RenderFrame {
            self.env.get_rf()
        }

        fn update(&mut self) {
            self.is_ticking = false;
        }

        fn tick(&mut self, amount: usize) -> Option<impl Future<Output = Result<(), TickError>>> {
            if self.is_ticking {
                return None;
            }

            self.is_ticking = true;

            for _ in 0..amount {
                self.env.tick();
            }

            Some(async move {
                tokio::task::spawn_blocking(move || for _ in 0..amount {})
                    .await
                    .map_err(|_| TickError::JoinFailed)
            })
        }
    }

    pub struct EnvironmentProxy {
        env: crate::Environment,
    }

    impl EnvironmentProxy {
        pub fn new() -> Self {
            let c = crate::Client::new("http://127.0.0.1", 40004);
            let kwargs = std::collections::HashMap::<&str, serde_json::Value>::from([(
                "render_mode",
                serde_json::to_value("rgb_array").unwrap(),
            )]);
            let env = c.make_env("MountainCar-v0", None, None, None, &kwargs);
            env.reset(None);

            Self { env }
        }

        pub fn tick(&self) {
            let action = self.env.action_space_sample();
            _ = self.env.step(&action);
        }

        pub fn get_rf(&self) -> crate::RenderFrame {
            self.env.render()
        }
    }
}
