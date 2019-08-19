use super::Coord;
use std::time::Instant;
use winit::{dpi::LogicalSize, *};

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

        output.map(|window| Self {
            events_loop,
            window,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<Coord<f32>>,
    pub new_mouse_position: Option<Coord<f32>>,
    pub seconds: f32,
}
impl UserInput {
    pub fn poll_events_loop(events_loop: &mut EventsLoop, last_timestamp: &mut Instant) -> Self {
        let mut output = UserInput::default();
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                output.end_requested = true;
                debug!("End was requested!");
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                output.new_frame_size =
                    Some(Coord::new(logical.width as f32, logical.height as f32));
                debug!("Our new frame size is {:?}", output.new_frame_size);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                output.new_mouse_position = Some(Coord::new(position.x as f32, position.y as f32));
            }
            _ => (),
        });

        output.seconds = {
            let now = Instant::now();
            let duration = now.duration_since(*last_timestamp);
            *last_timestamp = now;
            duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9
        };

        output
    }
}
