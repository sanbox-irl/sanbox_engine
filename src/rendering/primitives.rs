#[derive(Debug, Clone, Copy)]
pub struct Coord<T> {
    pub x: T,
    pub y: T,
}

impl<T> Coord<T> {
    pub fn new(x: T, y: T) -> Coord<T> {
        Coord { x, y }
    }
}
