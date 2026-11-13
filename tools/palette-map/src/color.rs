//! sRGB RGBA -> internal linear, premultiplied color space and the exact
//! perceptual distance used by libimagequant (`f_pixel::diff`).
//!
//! Keeping the distance identical to the host library means palette mapping
//! results match what the game engine would produce with the same palette.

/// 8-bit sRGB RGBA color. Layout matches `lodepng::RGBA` / `rgb::Rgba<u8>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Rgba8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba8 {
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[inline]
    pub fn is_fully_transparent(self) -> bool {
        self.a == 0
    }
}

impl From<[u8; 4]> for Rgba8 {
    #[inline]
    fn from(v: [u8; 4]) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }
}

impl From<Rgba8> for [u8; 4] {
    #[inline]
    fn from(c: Rgba8) -> Self {
        [c.r, c.g, c.b, c.a]
    }
}

// Internal weights, copied verbatim from imagequant/src/pal.rs so that the
// distance scale is interchangeable with the library's internal MSE.
const INTERNAL_GAMMA: f64 = 0.57;
const WEIGHT_A: f32 = 0.625;
const WEIGHT_R: f32 = 0.5;
const WEIGHT_G: f32 = 1.0;
const WEIGHT_B: f32 = 0.45;
const LIQ_WEIGHT_MSE: f64 = 0.45;

/// Linear, premultiplied, channel-weighted 4D point (a, r, g, b).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FPixel {
    pub a: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl FPixel {
    /// Build the sRGB gamma lookup table used by libimagequant.
    ///
    /// `gamma == 0.` means sRGB, exactly like
    /// `Attributes::new_image(..., 0.0)`.
    pub fn gamma_lut(gamma: f64) -> [f32; 256] {
        let gamma = if gamma > 0.0 { gamma } else { 0.45455 };
        let mut lut = [0.0; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = ((i as f32) / 255.0).powf((INTERNAL_GAMMA / gamma) as f32);
        }
        lut
    }

    /// Convert an 8-bit sRGB color into the internal weighted point.
    #[inline]
    pub fn from_rgba(lut: &[f32; 256], px: Rgba8) -> Self {
        let alpha = f32::from(px.a) / 255.0;
        Self {
            a: alpha * WEIGHT_A,
            r: lut[px.r as usize] * WEIGHT_R * alpha,
            g: lut[px.g as usize] * WEIGHT_G * alpha,
            b: lut[px.b as usize] * WEIGHT_B * alpha,
        }
    }

    /// Squared perceptual distance, identical to `f_pixel::diff`.
    ///
    /// For each RGB channel the maximum of the "on black" and "on white"
    /// premultiplied difference is used, so semi-transparent edge pixels are
    /// compared against both possible backgrounds instead of producing fringe
    /// artifacts. The alpha channel itself is not summed.
    #[inline]
    pub fn diff(self, other: FPixel) -> f32 {
        let alpha_diff = other.a - self.a;
        let black_r = self.r - other.r;
        let black_g = self.g - other.g;
        let black_b = self.b - other.b;
        let white_r = black_r + alpha_diff;
        let white_g = black_g + alpha_diff;
        let white_b = black_b + alpha_diff;
        (black_r * black_r).max(white_r * white_r)
            + (black_g * black_g).max(white_g * white_g)
            + (black_b * black_b).max(white_b * white_b)
    }
}

/// Convert internal squared distance into the equivalent standard MSE on
/// 0..255 channels (mean of the three RGB squared channel errors).
#[inline]
pub fn internal_dist_to_rmse(dist_sq: f64) -> f64 {
    (dist_sq * 65536.0 / 6.0 / LIQ_WEIGHT_MSE).sqrt()
}

/// Inverse of [`internal_dist_to_rmse`]: turn an RMSE threshold into the
/// internal squared-distance scale used by [`FPixel::diff`].
#[inline]
pub fn rmse_to_internal_dist(rmse: f64) -> f64 {
    let mse = rmse * rmse;
    mse * 6.0 * LIQ_WEIGHT_MSE / 65536.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_distance_for_identical_colors() {
        let lut = FPixel::gamma_lut(0.0);
        let p = FPixel::from_rgba(&lut, Rgba8::new(12, 200, 77, 255));
        assert_eq!(p.diff(p), 0.0);
    }

    #[test]
    fn distance_is_symmetric() {
        let lut = FPixel::gamma_lut(0.0);
        let x = FPixel::from_rgba(&lut, Rgba8::new(10, 20, 30, 200));
        let y = FPixel::from_rgba(&lut, Rgba8::new(200, 10, 90, 40));
        assert!((x.diff(y) - y.diff(x)).abs() < 1e-6);
    }

    #[test]
    fn rmse_roundtrip() {
        let original = 0.037;
        let back = internal_dist_to_rmse(rmse_to_internal_dist(original));
        assert!((back - original).abs() < 1e-9);
    }

    #[test]
    fn transparent_input_sits_at_origin() {
        let lut = FPixel::gamma_lut(0.0);
        assert_eq!(
            FPixel::from_rgba(&lut, Rgba8::new(255, 0, 128, 0)),
            FPixel::default()
        );
    }
}
