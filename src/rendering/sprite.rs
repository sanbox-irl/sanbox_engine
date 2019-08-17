pub enum SpriteName {
    Zelda,
    Link,
}

// Look into this!
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

pub struct Sprite {
    pub name: &'static SpriteName,
    pub file: &'static [u8],
    pub texture: usize,
}
