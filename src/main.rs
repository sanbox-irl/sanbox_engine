extern crate winit;
#[macro_use]
extern crate log;
extern crate arrayvec;
extern crate env_logger;
extern crate gfx_hal;
extern crate image;

mod rendering;

use gfx_hal::window::Suboptimal;
use nalgebra_glm as glm;
use rendering::{Coord, DrawingError, Sprite, TypedRenderer, UserInput, WinitState, SPRITE_LIST};
use std::time::Instant;

const WINDOW_NAME: &str = "Hello World!";
const DEFAULT_WINDOW_SIZE: Coord<f32> = Coord { x: 500.0, y: 800.0 };

fn main() {
    env_logger::init();
    let mut local_state = LocalState::new(DEFAULT_WINDOW_SIZE);
    let mut window_state =
        WinitState::new(WINDOW_NAME, DEFAULT_WINDOW_SIZE).expect("Error on windows creation.");
    let (mut renderer, mut sprites) = TypedRenderer::typed_new(
        &window_state.window,
        WINDOW_NAME,
        &SPRITE_LIST,
        &local_state,
    )
    .unwrap();

    let mut clean_exit = false;
    let mut time = Instant::now();

    loop {
        let inputs = UserInput::poll_events_loop(&mut window_state.events_loop, &mut time);
        if inputs.end_requested {
            clean_exit = true;
            break;
        }
        if inputs.new_frame_size.is_some() {
            debug!("Window changed size, creating a new swapchain...");
            if let Err(e) = renderer.recreate_swapchain(&window_state.window) {
                error!("Couldn't recreate the swapchain: {:?}", e);
                break;
            }

            local_state.frame_dimensions = inputs.new_frame_size.unwrap();
            for this_sprite in sprites.iter_mut() {
                this_sprite.update_window_scale(&local_state.frame_dimensions);
            }
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut renderer, &local_state, &sprites) {
            match e {
                DrawingError::AcquireAnImageFromSwapchain | DrawingError::PresentIntoSwapchain => {
                    debug!("Creating new swapchain!");
                    if let Err(e) = renderer.recreate_swapchain(&window_state.window) {
                        error!("Couldn't recreate the swapchain: {:?}", e);
                        break;
                    }
                }

                DrawingError::ResetFence | DrawingError::WaitOnFence => {
                    error!("Rendering Error: {:?}", e);
                    debug!("Auo-restarting Renderer...");
                    drop(renderer);
                    let ret = TypedRenderer::typed_new(
                        &window_state.window,
                        WINDOW_NAME,
                        &SPRITE_LIST,
                        &local_state,
                    );
                    match ret {
                        Ok(new_value) => {
                            renderer = new_value.0;
                            sprites = new_value.1;
                        }

                        Err(_) => {
                            error!("Couldn't recover from error.");
                            break;
                        }
                    }
                }
            }
        }
    }

    if clean_exit {
        info!("Exiting cleanly.");
    } else {
        error!("Exiting with error.");
    }
}

pub fn do_the_render(
    renderer: &mut TypedRenderer,
    _local_state: &LocalState,
    sprites: &Vec<Sprite>,
) -> Result<Option<Suboptimal>, DrawingError> {
    /*
    let x1 = 100.0;
    let y1 = 100.0;
    let quad1 = Quad {
        x: (x1 / local_state.frame_width as f32) * 2.0 - 1.0,
        y: (y1 / local_state.frame_height as f32) * 2.0 - 1.0,
        w: ((1280.0 - x1) / local_state.frame_width as f32) * 2.0,
        h: ((720.0 - y1) / local_state.frame_height as f32) * 2.0,
    };

    let quad2 = Quad {
        x: (200.0 / local_state.frame_width as f32) * 2.0 - 1.0,
        y: (200.0 / local_state.frame_height as f32) * 2.0 - 1.0,
        w: ((1280.0 - x1) / local_state.frame_width as f32) * 2.0,
        h: ((720.0 - y1) / local_state.frame_height as f32) * 2.0,
    };
    */
    let models = vec![
        // glm::translate(&glm::identity(), &glm::make_vec3(&[-1.0, 0.0, 0.0])),
        glm::identity(),
    ];

    renderer.draw_quad_frame(&models, &sprites)
}

#[derive(Debug)]
pub struct LocalState {
    pub frame_dimensions: Coord<f32>,
    pub mouse: Coord<f32>,
    pub spare_time: f32,
}
impl LocalState {
    pub fn new(frame_dimensions: Coord<f32>) -> LocalState {
        LocalState {
            frame_dimensions,
            mouse: Coord::new(0.0, 0.0),
            spare_time: 0.0,
        }
    }

    pub fn update_from_input(&mut self, input: UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_dimensions = frame_size;
        }
        if let Some(position) = input.new_mouse_position {
            self.mouse = position;
        }

        self.spare_time += input.seconds;
        const ONE_SIXTIETH: f32 = 1.0 / 60.0;

        while self.spare_time > 0.0 {
            self.spare_time -= ONE_SIXTIETH;
        }
    }
}
