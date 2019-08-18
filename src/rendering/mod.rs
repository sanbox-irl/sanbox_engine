macro_rules! manual_drop {
    ($this_val:expr) => {
        ManuallyDrop::into_inner(read(&$this_val))
    };
}

macro_rules! manual_new {
    ($this_val:ident) => {
        ManuallyDrop::new($this_val)
    };
}

mod buffer_bundle;
mod loaded_image;
mod primitives;
mod vertex;
mod renderer;
mod winit_state_user_input;
mod sprite;

pub use buffer_bundle::BufferBundle;
pub use loaded_image::LoadedImage;
pub use primitives::{Coord, Quad};
pub use renderer::TypedRenderer;
pub use winit_state_user_input::{UserInput, WinitState};
pub use sprite::*;
pub use vertex::*;