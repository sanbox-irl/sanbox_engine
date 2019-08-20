extern crate winit;
#[macro_use]
extern crate log;
extern crate arrayvec;
extern crate env_logger;
extern crate gfx_hal;
extern crate image;

mod rendering;

use arrayvec::ArrayVec;
use gfx_hal::window::Suboptimal;
use nalgebra_glm as glm;
use rendering::{Coord, DrawingError, Sprite, SpriteName, TypedRenderer, UserInput, WinitState, SPRITE_LIST};
use std::time::Instant;
use winit::{dpi::LogicalSize, VirtualKeyCode};

type TMat4f32 = glm::TMat4<f32>;

const WINDOW_NAME: &str = "Hello World!";

const WINDOW_SIZE_CYCLE: [Coord<f32>; 4] = [
    Coord { x: 640.0, y: 360.0 },
    Coord { x: 960.0, y: 540.0 },
    Coord { x: 1280.0, y: 720.0 },
    Coord { x: 1920.0, y: 1080.0 },
];

fn main() {
    env_logger::init();

    let mut screen_size = 0;

    let mut window_state = WinitState::new(WINDOW_NAME, WINDOW_SIZE_CYCLE[0]).expect("Error on windows creation.");
    let (mut renderer, mut sprites) =
        TypedRenderer::typed_new(&window_state.window, WINDOW_NAME, &SPRITE_LIST, &WINDOW_SIZE_CYCLE[0]).unwrap();
    let mut local_state = LocalState::new(WINDOW_SIZE_CYCLE[0]);
    let mut user_input = UserInput::default();

    local_state.entities.push(Entity::new(
        glm::make_vec3(&[1.0, 0.0, 0.0]),
        *sprites.get(&SpriteName::CenterDot).unwrap(),
    ));
    local_state.entities.push(Entity::new(
        glm::make_vec3(&[1.0, 0.0, 1.0]),
        *sprites.get(&SpriteName::Link).unwrap(),
    ));

    let mut clean_exit = false;
    let mut time = Instant::now();

    loop {
        user_input.poll_events_loop(&mut window_state.events_loop, &mut time);
        if user_input.end_requested {
            clean_exit = true;
            break;
        }
        if user_input.new_frame_size.is_some() {
            debug!("Window changed size, creating a new swapchain 0...");
            local_state.frame_dimensions = user_input.new_frame_size.unwrap();
            if let Err(e) = renderer.recreate_swapchain(&window_state.window) {
                error!("Couldn't recreate the swapchain: {:?}", e);
                break;
            }

            for this_sprite in sprites.iter_mut() {
                this_sprite.1.update_window_scale(&local_state.frame_dimensions);
            }
        }

        if user_input.pressed.contains(&VirtualKeyCode::Key1) {
            println!("Camera is looking like: {}", local_state.camera.position);
            println!("Camera Matrix is: {}", local_state.camera.make_view_matrix());

            let projection = {
                let mut temp = glm::ortho_lh_zo(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
                temp[(1, 1)] *= -1.0;
                temp
            };

            let view_projection = projection * local_state.camera.make_view_matrix();
            println!("View-Projection is {}", view_projection);
        }

        if user_input.pressed.contains(&VirtualKeyCode::Tab) {
            screen_size += 1;
            if screen_size > 3 {
                screen_size = 0;
            }

            window_state.window.set_inner_size(LogicalSize {
                width: WINDOW_SIZE_CYCLE[screen_size].x as f64,
                height: WINDOW_SIZE_CYCLE[screen_size].y as f64,
            });

            debug!("Window changed size, creating a new swapchain 1...");
            if let Err(e) = renderer.recreate_swapchain(&window_state.window) {
                error!("Couldn't recreate the swapchain: {:?}", e);
                break;
            }

            local_state.frame_dimensions = WINDOW_SIZE_CYCLE[screen_size];
            for this_sprite in sprites.iter_mut() {
                this_sprite.1.update_window_scale(&local_state.frame_dimensions);
            }
        }

        local_state.update_from_input(&user_input);
        if let Err(e) = do_the_render(&mut renderer, &local_state, &user_input) {
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
                        &WINDOW_SIZE_CYCLE[screen_size],
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
    local_state: &LocalState,
    user_input: &UserInput,
) -> Result<Option<Suboptimal>, DrawingError> {
    let projection = {
        let mut temp = glm::ortho_lh_zo(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        temp[(1, 1)] *= -1.0;
        temp
    };

    let view_projection = projection * local_state.camera.make_view_matrix();
    renderer.draw_quad_frame(
        &local_state.entities,
        &view_projection,
        user_input.pressed.contains(&VirtualKeyCode::Key1),
    )
}

#[derive(Debug)]
pub struct LocalState {
    pub frame_dimensions: Coord<f32>,
    pub mouse: Coord<f32>,
    pub spare_time: f32,
    pub camera: Camera,
    pub entities: ArrayVec<[Entity; 10]>,
}
impl LocalState {
    pub fn new(frame_dimensions: Coord<f32>) -> LocalState {
        LocalState {
            frame_dimensions,
            mouse: Coord::new(0.0, 0.0),
            spare_time: 0.0,
            camera: Camera::new_at_position(glm::make_vec3(&[0.0, 0.0, -1.0])),
            entities: ArrayVec::new(),
        }
    }

    pub fn update_from_input(&mut self, input: &UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_dimensions = frame_size;
        }
        if let Some(position) = input.new_mouse_position {
            self.mouse = position;
        }

        self.camera.update_position(input, 0.05);
        trace!("Camera position is {:?}", self.camera.position);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: glm::TVec3<f32>,
}

impl Camera {
    pub fn new_at_position(position: glm::TVec3<f32>) -> Camera {
        Camera { position }
    }

    pub fn update_position(&mut self, input: &UserInput, distance: f32) {
        let right: glm::TVec3<f32> = glm::make_vec3(&[1.0, 0.0, 0.0]);
        let up: glm::TVec3<f32> = glm::make_vec3(&[0.0, 1.0, 0.0]);

        let mut move_vector = input
            .held
            .iter()
            .fold(glm::make_vec3(&[0.0, 0.0, 0.0]), |vec, key| match *key {
                VirtualKeyCode::W | VirtualKeyCode::Up => vec + up,
                VirtualKeyCode::S | VirtualKeyCode::Down => vec - up,
                VirtualKeyCode::D | VirtualKeyCode::Right => vec + right,
                VirtualKeyCode::A | VirtualKeyCode::Left => vec - right,
                _ => vec,
            });
        if move_vector != glm::zero() {
            move_vector = move_vector.normalize();
        }
        self.position += move_vector * distance;
    }

    pub fn make_view_matrix(&self) -> TMat4f32 {
        glm::look_at_lh(
            &self.position,
            &glm::make_vec3(&[self.position[0], self.position[1], 0.0]),
            &glm::make_vec3(&[0.0, 1.0, 0.0]).normalize(),
        )
    }
}

#[derive(Debug)]
pub struct Entity {
    pub position: glm::TMat4<f32>,
    pub sprite: Sprite,
}

impl Entity {
    pub fn new(starting_position: glm::TVec3<f32>, sprite: Sprite) -> Self {
        Entity {
            position: glm::translate(&glm::identity(), &starting_position),
            sprite,
        }
    }
}
