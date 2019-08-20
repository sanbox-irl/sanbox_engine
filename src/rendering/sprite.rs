use super::Coord;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub enum SpriteName {
    Zelda,
    Link,
    CenterDot,
}

// Look into this! we want a macro that can handle this!
//sprites! [
//   Zelda => "../../resources/sprites/zelda.png",
//   Link => "../../resources/sprites/link.png",
//];
// 
pub const SPRITE_SIZE: usize = 3;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const SPRITE_LIST: [(SpriteName, &[u8]); SPRITE_SIZE] = [
    (
        SpriteName::Link,
        include_bytes!("../../resources/sprites/link.png"),
    ),
    (
        SpriteName::Zelda,
        include_bytes!("../../resources/sprites/zelda.png"),
    ),
    (
        SpriteName::CenterDot,
        include_bytes!("../../resources/sprites/center dot.png")
    )
];

#[derive(Debug, Copy, Clone)]
pub struct Sprite {
    pub name: SpriteName,
    pub texture_handle: usize,
    pub image_dimensions: Coord<u32>,
    internal_scaler: Coord<f32>,
}

use nalgebra_glm as glm;
impl Sprite {
    pub fn new(
        name: SpriteName,
        texture_handle: usize,
        image_dimensions: Coord<u32>,
        frame_dimensions: &Coord<f32>,
    ) -> Sprite {
        let mut ret = Sprite {
            name,
            texture_handle,
            image_dimensions,
            internal_scaler: Coord::new(0.0, 0.0),
        };
        ret.update_window_scale(frame_dimensions);

        ret
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

#[derive(Debug, Copy, Clone)]
pub struct Origin {
    pub horizontal: OriginHorizontal,
    pub vertical: OriginVertical,
}

impl Origin {
    pub fn new(horizontal: OriginHorizontal, vertical: OriginVertical) -> Origin {
        Origin { horizontal, vertical }
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
