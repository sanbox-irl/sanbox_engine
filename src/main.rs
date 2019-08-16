extern crate winit;
#[macro_use]
extern crate log;
extern crate arrayvec;
extern crate env_logger;
extern crate gfx_hal;
extern crate image;

mod rendering;

use gfx_hal::window::Suboptimal;
use rendering::{Coord, Quad, TypedRenderer, UserInput, WinitState};

const WINDOW_NAME: &str = "Hello World!";
const WINDOW_SIZE: Coord = Coord {
    x: 1920.0,
    y: 1080.0,
};

fn main() {
    env_logger::init();
    let mut window_state =
        WinitState::new(WINDOW_NAME, WINDOW_SIZE).expect("Error on windows creation.");
    let mut hal_state = TypedRenderer::typed_new(&window_state.window, WINDOW_NAME).unwrap();
    let mut local_state = LocalState::new(WINDOW_SIZE);

    loop {
        let inputs = UserInput::poll_events_loop(&mut window_state.events_loop);
        if inputs.end_requested {
            break;
        }
        if inputs.new_frame_size.is_some() {
            debug!("Window changed size, restarting Renderer...");
            drop(hal_state);

            hal_state = TypedRenderer::typed_new(&window_state.window, WINDOW_NAME).unwrap();
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut hal_state, &local_state) {
            error!("Rendering Error: {:?}", e);
            debug!("Auto-restarting Renderer...");
            drop(hal_state);
            hal_state = TypedRenderer::typed_new(&window_state.window, WINDOW_NAME).unwrap();
        }
    }
}

pub fn do_the_render(
    hal_state: &mut TypedRenderer,
    local_state: &LocalState,
) -> Result<Option<Suboptimal>, &'static str> {
    let x1 = 100.0;
    let y1 = 100.0;
    let quad = Quad {
        x: (x1 / local_state.frame_width as f32) * 2.0 - 1.0,
        y: (y1 / local_state.frame_height as f32) * 2.0 - 1.0,
        w: ((1280.0 - x1) / local_state.frame_width as f32) * 2.0,
        h: ((720.0 - y1) / local_state.frame_height as f32) * 2.0,
    };
    hal_state.draw_quad_frame(quad)
}

#[derive(Debug)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
}
impl LocalState {
    pub fn new(coord: Coord) -> LocalState {
        LocalState {
            frame_width: coord.x,
            frame_height: coord.y,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    pub fn update_from_input(&mut self, input: UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_width = frame_size.0;
            self.frame_height = frame_size.1;
        }
        if let Some(position) = input.new_mouse_position {
            self.mouse_x = position.0;
            self.mouse_y = position.1;
        }
    }
}
