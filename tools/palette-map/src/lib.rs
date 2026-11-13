//! `palette-map`: remap game textures onto a fixed external palette.
//!
//! Guarantees:
//! * output pixels only ever reference colors from the supplied palette;
//! * pixels farther than the configured threshold fall back to transparency
//!   or an explicitly named palette color;
//! * identical input always produces identical, pixel-exact output;
//! * the perceptual distance is premultiplied-alpha aware, so texture edges
//!   do not gain fringe/halo colors.

pub mod batch;
pub mod color;
pub mod fixture;
pub mod mapper;
pub mod nearest;
pub mod palette;

pub use color::{internal_dist_to_rmse, rmse_to_internal_dist, FPixel, Rgba8};
pub use mapper::{Fallback, ImageStats, MapConfig, PaletteMapper};
pub use nearest::NearestTree;
pub use palette::{load_palette, PaletteError, MAX_PALETTE_SIZE};
