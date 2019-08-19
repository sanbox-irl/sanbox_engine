use super::Coord;

#[derive(Debug)]
pub enum SpriteName {
    Zelda,
    Link,
}

// Look into this! we want a macro that can handle this!
//sprites! [
//   Zelda => "../../resources/sprites/zelda.png",
//   Link => "../../resources/sprites/link.png",
//];

pub static SPRITE_LIST: [(SpriteName, &[u8]); 2] = [
    (
        SpriteName::Zelda,
        include_bytes!("../../resources/sprites/zelda.png"),
    ),
    (
        SpriteName::Link,
        include_bytes!("../../resources/sprites/link.png"),
    ),
];

#[derive(Debug)]
pub struct Sprite {
    pub name: &'static SpriteName,
    pub file_bits: FileBits,
    pub texture_handle: usize,
    pub image_dimensions: Coord<u32>,
    internal_scaler: Coord<f32>,
}

use nalgebra_glm as glm;
impl Sprite {
    pub fn new(
        name: &'static SpriteName,
        file_bits: FileBits,
        texture_handle: usize,
        image_dimensions: Coord<u32>,
        frame_dimensions: &Coord<f32>,
    ) -> Sprite {
        Sprite {
            name,
            file_bits,
            texture_handle,
            image_dimensions,
            internal_scaler: Coord::new(
                2.0 * image_dimensions.x as f32 / (frame_dimensions.x),
                2.0 * image_dimensions.y as f32 / (frame_dimensions.y),
            ),
        }
    }

    pub fn update_window_scale(&mut self, frame_dimensions: &Coord<f32>) {
        self.internal_scaler = Coord::new(
            2.0 * self.image_dimensions.x as f32 / frame_dimensions.x,
            2.0 * self.image_dimensions.y as f32 / frame_dimensions.y,
        );
    }

    pub fn scale_by_sprite(&self, matrix: &glm::TMat4x4<f32>, origin: Origin) -> glm::TMat4x4<f32> {
        let scale = glm::make_vec3(&[self.internal_scaler.x, self.internal_scaler.y, 1.0]);
        let image_scaled = glm::scale(matrix, &scale);
        origin.translate(&image_scaled)
    }
}

pub struct FileBits(pub &'static [u8]);
impl std::fmt::Debug for FileBits {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "--")
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Origin {
    pub horizontal: OriginHorizontal,
    pub vertical: OriginVertical,
}

impl Origin {
    pub fn new(horizontal: OriginHorizontal, vertical: OriginVertical) -> Origin {
        Origin {
            horizontal,
            vertical,
        }
    }

    pub fn translate(&self, matrix: &glm::TMat4x4<f32>) -> glm::TMat4x4<f32> {
        glm::translate(
            matrix,
            &glm::make_vec3(&[self.horizontal.translate(), self.vertical.translate(), 0.0]),
        )
    }
}

#[derive(Debug, Copy, Clone)]
pub enum OriginHorizontal {
    Left,
    Center,
    Right,
    Custom(f32),
}

impl OriginHorizontal {
    fn translate(&self) -> f32 {
        match self {
            OriginHorizontal::Left => 0.0,
            OriginHorizontal::Center => -0.5,
            OriginHorizontal::Right => -1.0,
            OriginHorizontal::Custom(pixel_coord) => *pixel_coord,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum OriginVertical {
    Top,
    Center,
    Bottom,
    Custom(f32),
}

impl OriginVertical {
    fn translate(&self) -> f32 {
        match self {
            OriginVertical::Top => 0.0,
            OriginVertical::Center => -0.5,
            OriginVertical::Bottom => -1.0,
            OriginVertical::Custom(pixel_coord) => *pixel_coord,
        }
    }
}
