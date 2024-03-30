use super::policy::*;
use display::*;
use iced::executor;
use iced::theme::{self, Theme};
use iced::time;
use iced::widget::{button, column, container, row, slider, text};
use iced::{Alignment, Application, Command, Element, Length, Settings, Subscription};
use std::time::Duration;

pub type Result = iced::Result;

pub struct GymnasiumApp {
    display: Display,
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
    Reset,
}

impl Application for GymnasiumApp {
    type Message = Message;
    type Theme = Theme;
    type Executor = executor::Default;
    type Flags = EnvironmentProxyFlags;

    fn new(flags: EnvironmentProxyFlags) -> (Self, Command<Message>) {
        (
            Self {
                display: Display::new(flags),
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
        format!("Gymnasium - {}", self.display.name())
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Display(message, version) => {
                if version == self.version {
                    self.display.update(message);
                }
            }
            Message::Tick | Message::Next => {
                self.queued_ticks = (self.queued_ticks + 1).min(self.speed);

                if let Some(task) = self.display.tick(self.queued_ticks) {
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
            Message::Reset => {
                self.display.reset();
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
        let controls = Self::view_controls(self.is_playing, selected_speed);

        let content = column![
            self.display
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

impl GymnasiumApp {
    pub fn run(
        api_url: &str,
        instance_id: &str,
        reset_seed: Option<usize>,
        policy: Box<dyn Policy>,
    ) -> iced::Result {
        tracing_subscriber::fmt::init();

        <Self as Application>::run(Settings {
            antialiasing: true,
            window: iced::window::Settings {
                position: iced::window::Position::Centered,
                size: iced::Size {
                    height: 500.,
                    width: 650.,
                },
                // TODO: icon.
                ..iced::window::Settings::default()
            },
            ..Settings::with_flags(EnvironmentProxyFlags {
                api_url: api_url.to_string(),
                instance_id: instance_id.to_string(),
                reset_seed,
                policy,
            })
        })
    }

    fn view_controls<'a>(is_playing: bool, speed: usize) -> Element<'a, Message> {
        let playback_controls = row![
            button(if is_playing { "Pause" } else { "Play" }).on_press(Message::TogglePlayback),
            button("Next")
                .on_press_maybe((!is_playing).then_some(Message::Next))
                .style(theme::Button::Secondary),
        ]
        .spacing(10);

        let speed_controls = row![
            slider(1.0..=40.0, speed as f32, Message::SpeedChanged),
            text(format!("x{speed}")).size(16),
        ]
        .align_items(Alignment::Center)
        .spacing(10);

        row![
            playback_controls,
            speed_controls,
            button("Reset")
                .on_press_maybe((!is_playing).then_some(Message::Reset))
                .style(theme::Button::Destructive)
        ]
        .padding(10)
        .spacing(20)
        .align_items(Alignment::Center)
        .into()
    }
}

pub mod display {
    use super::super::{policy::Policy, Environment, ObsActSpaceItem, RenderFrame};
    use base64::prelude::*;
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
        pub fn new(flags: EnvironmentProxyFlags) -> Self {
            let env = EnvironmentProxy::new(flags);

            Self {
                state: State::with_env(env),
                last_tick_duration: Duration::default(),
                last_queued_ticks: 0,
            }
        }

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

        pub fn reset(&mut self) {
            self.state.reset();
        }

        pub fn name(&self) -> &str {
            self.state.name()
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
            let rgb = self.state.render_frame();
            let rgb = rgb.as_rgb().unwrap();

            let bytes = BASE64_STANDARD.decode(rgb.2).unwrap();
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

        fn render_frame(&self) -> RenderFrame {
            self.env.render_frame()
        }

        pub fn reset(&mut self) {
            self.env.reset();
        }

        pub fn name(&self) -> &str {
            self.env.name()
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
                tokio::task::spawn_blocking(move || ())
                    .await
                    .map_err(|_| TickError::JoinFailed)
            })
        }
    }

    pub struct EnvironmentProxyFlags {
        pub api_url: String,
        pub instance_id: String,
        pub reset_seed: Option<usize>,
        pub policy: Box<dyn Policy>,
    }

    pub struct EnvironmentProxy {
        env: Environment,
        env_name: String,
        reset_seed: Option<usize>,
        last_known_state: Vec<ObsActSpaceItem>,
        policy: Box<dyn Policy>,
    }

    impl EnvironmentProxy {
        pub fn new(flags: EnvironmentProxyFlags) -> Self {
            let env = Environment::reference(&flags.api_url, &flags.instance_id);
            let last_known_state = env.reset(flags.reset_seed);
            let env_name = env.name();

            Self {
                env,
                env_name,
                reset_seed: flags.reset_seed,
                last_known_state,
                policy: flags.policy,
            }
        }

        pub fn tick(&mut self) {
            let action = self.policy.policy(&self.last_known_state);
            let si = self.env.step(&action);
            self.last_known_state = si.observation;
        }

        /// PERF: 2 of these are getting called for every step.
        pub fn render_frame(&self) -> RenderFrame {
            self.env.render()
        }

        pub fn reset(&mut self) {
            self.last_known_state = self.env.reset(self.reset_seed);
        }

        pub fn name(&self) -> &str {
            &self.env_name
        }
    }
}
