#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub points: [[f32; 2]; 3],
}
impl Triangle {
    #[allow(dead_code)]
    pub fn vertex_attributes(self) -> [f32; 3 * (2 + 3)] {
        let [[a, b], [c, d], [e, f]] = self.points;
        [
            a, b, 1.0, 0.0, 0.0, // red
            c, d, 0.0, 1.0, 0.0, // green
            e, f, 0.0, 0.0, 1.0, // blue
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Quad {
    pub fn vertex_attributes(self) -> [f32; 4 * (2 + 3 + 2)] {
        let x = self.x;
        let y = self.y;
        let w = self.w;
        let h = self.h;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        [
            // X    Y       R       G       B       type        X       Y       human location
            x,      y + h,  1.0,    0.0,    0.0, /* red   */    0.0,    1.0, /* bottom left */
            x,      y,      0.0,    1.0,    0.0, /* green */    0.0,    0.0, /* top left*/
            x + w,  y,      0.0,    0.0,    1.0, /* blue  */    1.0,    0.0, /* top right */
            x + w,  y + h,  1.0,    0.0,    1.0, /* magenta*/   1.0,    1.0, /* bottom right */
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    pub fn new(x: f64, y: f64) -> Coord {
        Coord { x, y }
    }
}
