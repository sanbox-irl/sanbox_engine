use super::Coord;
use arrayvec::ArrayVec;
use std::time::Instant;
use winit::{
    dpi::LogicalSize, CreationError, DeviceEvent, ElementState, Event, EventsLoop, KeyboardInput, VirtualKeyCode,
    Window, WindowBuilder, WindowEvent,
};

pub struct WinitState {
    pub events_loop: EventsLoop,
    pub window: Window,
}
impl WinitState {
    pub fn new<T: Into<String>>(title: T, coord: Coord<f32>) -> Result<Self, CreationError> {
        let events_loop = EventsLoop::new();
        let output = WindowBuilder::new()
            .with_title(title)
            .with_dimensions(LogicalSize {
                width: coord.x as f64,
                height: coord.y as f64,
            })
            .build(&events_loop);

        output.map(|window| Self { events_loop, window })
    }
}

#[derive(Debug, Clone)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<Coord<f32>>,
    pub new_mouse_position: Option<Coord<f32>>,
    pub seconds: f32,
    pub pressed: ArrayVec<[VirtualKeyCode; 20]>,
    pub held: ArrayVec<[VirtualKeyCode; 20]>,
    pub released: ArrayVec<[VirtualKeyCode; 20]>,
}

impl Default for UserInput {
    fn default() -> Self {
        UserInput {
            end_requested: false,
            new_frame_size: None,
            new_mouse_position: None,
            seconds: 0.0,
            pressed: ArrayVec::new(),
            held: ArrayVec::new(),
            released: ArrayVec::new(),
        }
    }
}

impl UserInput {
    pub fn poll_events_loop(&mut self, events_loop: &mut EventsLoop, last_timestamp: &mut Instant) {
        // Save our Pressed last frame...
        let last_frame_pressed = self.pressed.clone();
        self.clear_input();

        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                self.end_requested = true;
                debug!("End was requested!");
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                self.new_frame_size = Some(Coord::new(logical.width as f32, logical.height as f32));
                debug!("Our new frame size is {:?}", self.new_frame_size);
            }

            Event::DeviceEvent {
                event:
                    DeviceEvent::Key(KeyboardInput {
                        virtual_keycode: Some(code),
                        state,
                        ..
                    }),
                ..
            } => self.record_input(state, code, &last_frame_pressed),

            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                self.new_mouse_position = Some(Coord::new(position.x as f32, position.y as f32));
            }

            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(code),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                if cfg!(feature = "metal") {
                    self.record_input(state, code, &last_frame_pressed);
                }
            }
            _ => (),
        });

        self.seconds = {
            let now = Instant::now();
            let duration = now.duration_since(*last_timestamp);
            *last_timestamp = now;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };
    }

    pub fn record_input(
        &mut self,
        element_state: ElementState,
        code: VirtualKeyCode,
        last_frame_pressed: &[VirtualKeyCode],
    ) {
        match element_state {
            ElementState::Pressed => {
                if let None = last_frame_pressed.iter().position(|&pos| pos == code) {
                    if let None = self.held.iter().position(|&pos| pos == code) {
                        trace!("Pressed key {:?}", code);
                        self.pressed.push(code);
                        self.held.push(code);
                    }
                }
            }

            ElementState::Released => {
                if let Some(vk_pos) = self.held.iter().position(|&item| item == code) {
                    self.held.remove(vk_pos);
                    self.released.push(code);
                }
            }
        }
    }

    fn clear_input(&mut self) {
        self.pressed.clear();
        self.released.clear();

        self.end_requested = false;
        self.new_frame_size = None;
        self.new_mouse_position = None;
    }
}
