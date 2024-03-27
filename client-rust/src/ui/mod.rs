use grid::Grid;

use iced::executor;
use iced::theme::{self, Theme};
use iced::time;
use iced::widget::{button, column, container, row, slider, text};
use iced::{Alignment, Command, Element, Length, Subscription};
use std::time::Duration;

pub type Result = iced::Result;

#[derive(Default)]
pub struct Application {
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
    Clear,
    SpeedChanged(f32),
}

impl iced::Application for Application {
    type Message = Message;
    type Theme = Theme;
    type Executor = executor::Default;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        (
            Self {
                speed: 5,
                ..Self::default()
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
            Message::Clear => {
                self.grid.clear();
                self.version += 1;
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

    row![
        playback_controls,
        speed_controls,
        button("Clear")
            .on_press(Message::Clear)
            .style(theme::Button::Destructive),
    ]
    .padding(10)
    .spacing(20)
    .align_items(Alignment::Center)
    .into()
}

mod grid {
    use iced::alignment;
    use iced::mouse;
    use iced::widget::canvas;
    use iced::widget::canvas::event::{self, Event};
    use iced::widget::canvas::{Cache, Canvas, Frame, Geometry, Path, Text};
    use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Theme, Vector};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::future::Future;
    use std::ops::RangeInclusive;
    use std::time::{Duration, Instant};

    pub struct Grid {
        state: State,
        life_cache: Cache,
        grid_cache: Cache,
        translation: Vector,
        scaling: f32,
        last_tick_duration: Duration,
        last_queued_ticks: usize,
    }

    #[derive(Debug, Clone)]
    pub enum Message {
        Populate(Cell),
        Unpopulate(Cell),
        Translated(Vector),
        Scaled(f32, Option<Vector>),
        Ticked {
            result: Result<Life, TickError>,
            tick_duration: Duration,
        },
    }

    #[derive(Debug, Clone)]
    pub enum TickError {
        JoinFailed,
    }

    impl Default for Grid {
        fn default() -> Self {
            Self::from_preset()
        }
    }

    impl Grid {
        pub fn from_preset() -> Self {
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
                state: State::with_life(life.into_iter().map(|(i, j)| Cell { i, j }).collect()),
                life_cache: Cache::default(),
                grid_cache: Cache::default(),
                translation: Vector::default(),
                scaling: 1.0,
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
                Message::Scaled(scaling, translation) => {
                    self.scaling = scaling;

                    if let Some(translation) = translation {
                        self.translation = translation;
                    }

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
            Canvas::new(self)
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        }

        pub fn clear(&mut self) {
            self.state = State::default();

            self.life_cache.clear();
        }

        fn visible_region(&self, size: Size) -> Region {
            let width = size.width / self.scaling;
            let height = size.height / self.scaling;

            Region {
                x: -self.translation.x - width / 2.0,
                y: -self.translation.y - height / 2.0,
                width,
                height,
            }
        }
    }

    impl canvas::Program<Message> for Grid {
        type State = Interaction;

        fn update(
            &self,
            _interaction: &mut Interaction,
            _event: Event,
            _bounds: Rectangle,
            _cursor: mouse::Cursor,
        ) -> (event::Status, Option<Message>) {
            (event::Status::Ignored, None)
        }

        fn draw(
            &self,
            _interaction: &Interaction,
            renderer: &Renderer,
            _theme: &Theme,
            bounds: Rectangle,
            _cursor: mouse::Cursor,
        ) -> Vec<Geometry> {
            let center = Vector::new(bounds.width / 2.0, bounds.height / 2.0);

            let life = self.life_cache.draw(renderer, bounds.size(), |frame| {
                let background = Path::rectangle(Point::ORIGIN, frame.size());
                frame.fill(&background, Color::from_rgb8(0x40, 0x44, 0x4B));

                frame.with_save(|frame| {
                    frame.translate(center);
                    frame.scale(self.scaling);
                    frame.translate(self.translation);
                    frame.scale(Cell::SIZE);

                    let region = self.visible_region(frame.size());

                    for cell in region.cull(self.state.cells()) {
                        frame.fill_rectangle(
                            Point::new(cell.j as f32, cell.i as f32),
                            Size::UNIT,
                            Color::WHITE,
                        );
                    }
                });
            });

            let overlay = {
                let mut frame = Frame::new(renderer, bounds.size());

                let text = Text {
                    color: Color::WHITE,
                    size: 14.0.into(),
                    position: Point::new(0., 0.),
                    horizontal_alignment: alignment::Horizontal::Left,
                    vertical_alignment: alignment::Vertical::Top,
                    ..Text::default()
                };

                let cell_count = self.state.cell_count();

                frame.fill_text(Text {
                    content: format!(
                        "{cell_count} cell{} @ {:?} ({})",
                        if cell_count == 1 { "" } else { "s" },
                        self.last_tick_duration,
                        self.last_queued_ticks
                    ),
                    ..text
                });

                frame.into_geometry()
            };

            vec![life, overlay]
        }
    }

    #[derive(Default)]
    struct State {
        life: Life,
        births: FxHashSet<Cell>,
        is_ticking: bool,
    }

    impl State {
        pub fn with_life(life: Life) -> Self {
            Self {
                life,
                ..Self::default()
            }
        }

        fn cell_count(&self) -> usize {
            self.life.len() + self.births.len()
        }

        fn cells(&self) -> impl Iterator<Item = &Cell> {
            self.life.iter().chain(self.births.iter())
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

    #[derive(Clone, Default)]
    pub struct Life {
        cells: FxHashSet<Cell>,
    }

    impl Life {
        fn len(&self) -> usize {
            self.cells.len()
        }

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
        const SIZE: u16 = 20;

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

    pub struct Region {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    }

    impl Region {
        fn rows(&self) -> RangeInclusive<isize> {
            let first_row = (self.y / Cell::SIZE as f32).floor() as isize;

            let visible_rows = (self.height / Cell::SIZE as f32).ceil() as isize;

            first_row..=first_row + visible_rows
        }

        fn columns(&self) -> RangeInclusive<isize> {
            let first_column = (self.x / Cell::SIZE as f32).floor() as isize;

            let visible_columns = (self.width / Cell::SIZE as f32).ceil() as isize;

            first_column..=first_column + visible_columns
        }

        fn cull<'a>(
            &self,
            cells: impl Iterator<Item = &'a Cell>,
        ) -> impl Iterator<Item = &'a Cell> {
            let rows = self.rows();
            let columns = self.columns();

            cells.filter(move |cell| rows.contains(&cell.i) && columns.contains(&cell.j))
        }
    }

    pub enum Interaction {
        None,
    }

    impl Default for Interaction {
        fn default() -> Self {
            Self::None
        }
    }
}
