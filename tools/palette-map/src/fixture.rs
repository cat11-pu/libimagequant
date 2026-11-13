//! Deterministic synthetic fixture generator for end-to-end batch runs.
//!
//! Produces one 256-entry external palette and 500 RGBA "character skin"
//! textures. Textures deliberately include semi-transparent anti-aliased
//! edges, gradients and off-palette noise so that fallback, edge handling and
//! statistics all have something to do.

use crate::batch::save_rgba_png;
use crate::color::Rgba8;
use std::path::Path;

/// Build the fixed 256-color palette.
///
/// Layout (all entries are distinct and in a stable order):
/// * 1 fully transparent entry;
/// * 160 opaque RGB grid colors (4 x 8 x 5);
/// * 31 opaque grayscale steps;
/// * 32 semi-transparent edge colors (4 skin tones x 8 alpha steps), so
///   anti-aliased sprite edges have legal palette targets instead of being
///   pushed onto the fully transparent entry;
/// * 32 opaque skin-tone fills.
///
/// Total: 1 + 160 + 31 + 32 + 32 = 256.
pub fn make_palette_256() -> Vec<Rgba8> {
    let mut colors = vec![Rgba8::new(0, 0, 0, 0)];

    // 4 x 8 x 5 = 160 structured opaque grid colors.
    for r_step in 0..4u8 {
        for g_step in 0..8u8 {
            for b_step in 0..5u8 {
                colors.push(Rgba8::new(
                    r_step * 84,
                    g_step * 36,
                    b_step * 63,
                    255,
                ));
            }
        }
    }

    // 31 opaque grayscale steps (skip pure black to avoid a grid collision).
    for i in 1..=31u8 {
        let v = i * 8;
        colors.push(Rgba8::new(v, v, v, 255));
    }

    // 4 skin tones x 8 alpha steps = 32 semi-transparent edge colors.
    let skin_tones = [(210, 165, 122), (170, 120, 86), (232, 190, 150), (140, 96, 66)];
    for (r, g, b) in skin_tones {
        for step in 1..=8u8 {
            let alpha = step * 31; // 31..248, never 0 or 255
            colors.push(Rgba8::new(r, g, b, alpha));
        }
    }

    // 32 opaque fills chosen off the grid/gray lattice.
    for i in 0..32u8 {
        colors.push(Rgba8::new(
            180 + (i % 8) * 5,
            130 + (i / 8) * 6,
            90 + (i % 4) * 7,
            255,
        ));
    }

    debug_assert_eq!(colors.len(), 256);
    colors
}

/// Tiny deterministic PRNG (xorshift32) so fixture data is reproducible.
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    fn byte(&mut self) -> u8 {
        (self.next_u32() & 0xff) as u8
    }
}

/// Generate `count` 64x64 RGBA textures under `dir`, named
/// `skin_0000.png` ... in a sorted order.
pub fn generate_fixtures(dir: &Path, count: usize) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let size = 64usize;
    for image_index in 0..count {
        let mut rng = Rng::new(0x9e3779b9 ^ (image_index as u32).wrapping_mul(0x85ebca6b));
        let base_r = 60 + rng.byte() % 140;
        let base_g = 40 + rng.byte() % 110;
        let base_b = 30 + rng.byte() % 90;
        let radius = 18 + rng.byte() % 14;
        let center = size as f32 / 2.0;

        let mut pixels = vec![Rgba8::new(0, 0, 0, 0); size * size];
        for row in 0..size {
            for col in 0..size {
                let dx = col as f32 - center;
                let dy = row as f32 - center;
                let dist = (dx * dx + dy * dy).sqrt();
                let alpha = if dist < radius as f32 {
                    255
                } else if dist < radius as f32 + 1.5 {
                    // Anti-aliased edge band with partial alpha.
                    (((radius as f32 + 1.5 - dist) / 1.5) * 255.0) as u8
                } else {
                    0
                };

                // Skin shading gradient plus sparse off-palette noise.
                let shade = ((row as f32 / size as f32) * 60.0) as i32 - 30;
                // `is_multiple_of` is only stable since Rust 1.80; keep the
                // modulo form for the crate's 1.65 minimum toolchain.
                #[allow(clippy::manual_is_multiple_of)]
                let noisy = rng.next_u32() % 11 == 0;
                let noise = if noisy {
                    (rng.byte() % 40) as i32 - 20
                } else {
                    0
                };
                let clamp = |v: i32| v.clamp(0, 255) as u8;
                pixels[row * size + col] = Rgba8::new(
                    clamp(base_r as i32 + shade + noise),
                    clamp(base_g as i32 + shade / 2 + noise),
                    clamp(base_b as i32 + shade / 3 + noise),
                    alpha,
                );
            }
        }

        let name = format!("skin_{image_index:04}.png");
        save_rgba_png(&dir.join(name), &pixels, size, size)?;
    }
    Ok(())
}

/// Write the palette as a 16x16 PNG swatch in palette order.
pub fn write_palette_png(path: &Path, colors: &[Rgba8]) -> Result<(), String> {
    save_rgba_png(path, colors, 16, 16)
}
