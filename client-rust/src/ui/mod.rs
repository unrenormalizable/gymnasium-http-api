use grid::Grid;

use iced::executor;
use iced::theme::{self, Theme};
use iced::time;
use iced::widget::{button, column, container, row, slider, text};
use iced::{Alignment, Application, Command, Element, Length, Subscription};
use std::time::Duration;

pub type Result = iced::Result;

pub struct GymnasiumApp {
    grid: Grid,
    is_playing: bool,
    queued_ticks: usize,
    speed: usize,
    next_speed: Option<usize>,
    version: usize,
}

#[derive(Debug, Clone)]
pub enum Message {
    Grid(grid::Message, usize),
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
                grid: Grid::new(),
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
        String::from("Gymnasium - ???")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Grid(message, version) => {
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

                    return Command::perform(task, move |message| Message::Grid(message, version));
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
        let controls = view_controls(
            self.is_playing,
            selected_speed,
        );

        let content = column![
            self.grid
                .view()
                .map(move |message| Message::Grid(message, version)),
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

fn view_controls<'a>(
    is_playing: bool,
    speed: usize,
) -> Element<'a, Message> {
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

    row![
        playback_controls,
        speed_controls,
    ]
    .padding(10)
    .spacing(20)
    .align_items(Alignment::Center)
    .into()
}

mod grid {
    use iced::widget::canvas::Cache;
    use iced::{Element, Length, Vector};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::future::Future;
    use std::time::{Duration, Instant};

    pub struct Grid {
        state: State,
        life_cache: Cache,
        grid_cache: Cache,
        translation: Vector,
        last_tick_duration: Duration,
        last_queued_ticks: usize,
    }

    #[derive(Debug, Clone)]
    pub enum Message {
        Populate(Cell),
        Unpopulate(Cell),
        Translated(Vector),
        Ticked {
            result: Result<Life, TickError>,
            tick_duration: Duration,
        },
    }

    #[derive(Debug, Clone)]
    pub enum TickError {
        JoinFailed,
    }

    impl Grid {
        pub fn new() -> Self {
            #[rustfmt::skip]
            let cells = vec![
                "  xxx  ",
                "  x x  ",
                "  x x  ",
                "   x   ",
                "x xxx  ",
                " x x x ",
                "   x  x",
                "  x x  ",
                "  x x  ",
            ];

            let start_row = -(cells.len() as isize / 2);

            let life = cells
                .into_iter()
                .enumerate()
                .flat_map(|(i, cells)| {
                    let start_column = -(cells.len() as isize / 2);

                    cells
                        .chars()
                        .enumerate()
                        .filter(|(_, c)| !c.is_whitespace())
                        .map(move |(j, _)| (start_row + i as isize, start_column + j as isize))
                })
                .collect::<Vec<(isize, isize)>>();

            Self {
                state: State::with_life(EnvironmentProxy::new(), life.into_iter().map(|(i, j)| Cell { i, j }).collect()),
                life_cache: Cache::default(),
                grid_cache: Cache::default(),
                translation: Vector::default(),
                last_tick_duration: Duration::default(),
                last_queued_ticks: 0,
            }
        }
    }

    impl Grid {
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
                Message::Populate(cell) => {
                    self.state.populate(cell);
                    self.life_cache.clear();
                }
                Message::Unpopulate(cell) => {
                    self.state.unpopulate(&cell);
                    self.life_cache.clear();
                }
                Message::Translated(translation) => {
                    self.translation = translation;

                    self.life_cache.clear();
                    self.grid_cache.clear();
                }
                Message::Ticked {
                    result: Ok(life),
                    tick_duration,
                } => {
                    self.state.update(life);
                    self.life_cache.clear();

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

            let handle = iced::widget::image::Handle::from_pixels(*rgb.1 as u32, *rgb.0 as u32, bytes);
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
        life: Life,
        births: FxHashSet<Cell>,
        is_ticking: bool,
    }

    impl State {
        pub fn with_life(env: EnvironmentProxy, life: Life) -> Self {
            Self {
                env,
                life,
                births: Default::default(),
                is_ticking: Default::default(),
            }
        }

        fn get_rf(&self) -> crate::RenderFrame {
            self.env.get_rf()
        }

        fn populate(&mut self, cell: Cell) {
            if self.is_ticking {
                self.births.insert(cell);
            } else {
                self.life.populate(cell);
            }
        }

        fn unpopulate(&mut self, cell: &Cell) {
            if self.is_ticking {
                let _ = self.births.remove(cell);
            } else {
                self.life.unpopulate(cell);
            }
        }

        fn update(&mut self, mut life: Life) {
            self.births.drain().for_each(|cell| life.populate(cell));

            self.life = life;
            self.is_ticking = false;
        }

        fn tick(&mut self, amount: usize) -> Option<impl Future<Output = Result<Life, TickError>>> {
            if self.is_ticking {
                return None;
            }

            self.is_ticking = true;

            for _ in 0..amount {
                self.env.tick();
            }

            let mut life = self.life.clone();

            Some(async move {
                tokio::task::spawn_blocking(move || {
                    for _ in 0..amount {
                        life.tick();
                    }

                    life
                })
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
            let kwargs = std::collections::HashMap::<&str, serde_json::Value>::from([("render_mode", serde_json::to_value("rgb_array").unwrap())]);
            let env = c.make_env("MountainCar-v0", None, None, None, &kwargs);
            env.reset(None);

            Self {
                env,
            }
        }

        pub fn tick(&self) {
            let action = self.env.action_space_sample();
            _ = self.env.step(&action);
        }

        pub fn get_rf(&self) -> crate::RenderFrame {
            self.env.render()
        }
    }

    #[derive(Clone, Default)]
    pub struct Life {
        cells: FxHashSet<Cell>,
    }

    impl Life {
        fn populate(&mut self, cell: Cell) {
            self.cells.insert(cell);
        }

        fn unpopulate(&mut self, cell: &Cell) {
            let _ = self.cells.remove(cell);
        }

        fn tick(&mut self) {
            let mut adjacent_life = FxHashMap::default();

            for cell in &self.cells {
                let _ = adjacent_life.entry(*cell).or_insert(0);

                for neighbor in Cell::neighbors(*cell) {
                    let amount = adjacent_life.entry(neighbor).or_insert(0);

                    *amount += 1;
                }
            }

            for (cell, amount) in &adjacent_life {
                match amount {
                    2 => {}
                    3 => {
                        let _ = self.cells.insert(*cell);
                    }
                    _ => {
                        let _ = self.cells.remove(cell);
                    }
                }
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = &Cell> {
            self.cells.iter()
        }
    }

    impl std::iter::FromIterator<Cell> for Life {
        fn from_iter<I: IntoIterator<Item = Cell>>(iter: I) -> Self {
            Life {
                cells: iter.into_iter().collect(),
            }
        }
    }

    impl std::fmt::Debug for Life {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Life")
                .field("cells", &self.cells.len())
                .finish()
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Cell {
        i: isize,
        j: isize,
    }

    impl Cell {
        fn cluster(cell: Cell) -> impl Iterator<Item = Cell> {
            use itertools::Itertools;

            let rows = cell.i.saturating_sub(1)..=cell.i.saturating_add(1);
            let columns = cell.j.saturating_sub(1)..=cell.j.saturating_add(1);

            rows.cartesian_product(columns).map(|(i, j)| Cell { i, j })
        }

        fn neighbors(cell: Cell) -> impl Iterator<Item = Cell> {
            Cell::cluster(cell).filter(move |candidate| *candidate != cell)
        }
    }
}
