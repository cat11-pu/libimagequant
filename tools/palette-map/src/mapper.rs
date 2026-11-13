//! Fixed-palette pixel mapping with threshold fallback and per-image stats.

use crate::color::{internal_dist_to_rmse, rmse_to_internal_dist, FPixel, Rgba8};
use crate::nearest::NearestTree;

/// What happens when the closest palette entry is farther than the threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// Output a fully transparent pixel. Requires a transparent palette entry.
    Transparent,
    /// Output the given palette entry (by palette index).
    Color(u16),
}

#[derive(Clone, Debug)]
pub struct MapConfig {
    /// Maximum acceptable distance in 0..255-scale RMSE (RGB mean).
    /// `None` disables the threshold entirely.
    pub max_rmse: Option<f64>,
    pub fallback: Fallback,
    /// sRGB gamma; `0.0` selects the standard sRGB transfer curve.
    pub gamma: f64,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_rmse: None,
            fallback: Fallback::Transparent,
            gamma: 0.0,
        }
    }
}

/// Per-image mapping statistics. All distances are in 0..255-scale RMSE.
#[derive(Clone, Debug, Default)]
pub struct ImageStats {
    pub width: usize,
    pub height: usize,
    pub total_pixels: u64,
    pub transparent_input_pixels: u64,
    pub fallback_pixels: u64,
    /// Pixels whose input color matched a palette entry exactly.
    pub exact_matches: u64,
    /// Squared internal distances of every opaque/semi-opaque input pixel.
    pub(crate) distances: Vec<f32>,
    /// Worst non-fallback pixel distance found.
    pub max_dist_rmse: f64,
    /// Number of opaque/semi-opaque pixel neighbors that straddle a
    /// transparency boundary and were both mapped from high-error edge colors.
    pub edge_fringe_hits: u64,
}

impl ImageStats {
    fn record(&mut self, dist_sq: f32, exact: bool, fell_back: bool) {
        self.exact_matches += exact as u64;
        self.fallback_pixels += fell_back as u64;
        if fell_back {
            return;
        }
        let rmse = internal_dist_to_rmse(f64::from(dist_sq));
        if rmse > self.max_dist_rmse {
            self.max_dist_rmse = rmse;
        }
        self.distances.push(dist_sq);
    }

    /// Mean RMSE over non-fallback pixels.
    pub fn mean_rmse(&self) -> f64 {
        if self.distances.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .distances
            .iter()
            .map(|d| internal_dist_to_rmse(f64::from(*d)))
            .sum();
        sum / self.distances.len() as f64
    }

 /// RMSE at the requested percentile (`0..=100`) over non-fallback pixels.
    pub fn percentile_rmse(&self, percentile: f64) -> f64 {
        if self.distances.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.distances.clone();
        sorted.sort_by(f32::total_cmp);
        let rank = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        internal_dist_to_rmse(f64::from(sorted[rank.min(sorted.len() - 1)]))
    }

    pub fn mapped_pixels(&self) -> u64 {
        self.distances.len() as u64
    }
}

pub struct PaletteMapper {
    palette: Vec<Rgba8>,
    tree: NearestTree,
    lut: [f32; 256],
    threshold_sq: Option<f32>,
    fallback_index: u16,
    /// Fully transparent palette entries. These are valid targets for fully
    /// transparent input and for threshold fallback, but must never win the
    /// nearest-neighbor vote for an opaque/semi-opaque pixel, otherwise sprite
    /// edges get punched full of transparent holes (fringe artifacts).
    transparent_entries: Vec<u16>,
}

impl PaletteMapper {
    pub fn new(palette: Vec<Rgba8>, config: &MapConfig) -> Result<Self, String> {
        let lut = FPixel::gamma_lut(config.gamma);
        let points: Vec<FPixel> = palette
            .iter()
            .map(|c| FPixel::from_rgba(&lut, *c))
            .collect();
        let tree = NearestTree::build(points.clone()).map_err(str::to_string)?;

        let fallback_index = match config.fallback {
            Fallback::Transparent => palette
                .iter()
                .position(|c| c.is_fully_transparent())
                .ok_or_else(|| {
                    "fallback=transparent requires a fully transparent (a=0) palette entry"
                        .to_string()
                })? as u16,
            Fallback::Color(index) => {
                if usize::from(index) >= palette.len() {
                    return Err(format!(
                        "fallback palette index {index} is out of range (palette has {} entries)",
                        palette.len()
                    ));
                }
                index
            }
        };

        let threshold_sq = config
            .max_rmse
            .map(rmse_to_internal_dist)
            .map(|d| d as f32);

        let transparent_entries: Vec<u16> = palette
            .iter()
            .enumerate()
            .filter(|(_, c)| c.is_fully_transparent())
            .map(|(i, _)| i as u16)
            .collect();
        if transparent_entries.len() == palette.len() {
            return Err("palette must contain at least one non-transparent color".to_string());
        }

        Ok(Self {
            palette,
            tree,
            lut,
            threshold_sq,
            fallback_index,
            transparent_entries,
        })
    }

    pub fn palette(&self) -> &[Rgba8] {
        &self.palette
    }

    /// Map one RGBA image. Returns palette-index output and statistics.
    pub fn map(&self, pixels: &[Rgba8], width: usize, height: usize) -> (Vec<u8>, ImageStats) {
        debug_assert_eq!(pixels.len(), width * height);
        let mut indices = vec![0u8; pixels.len()];
        let mut stats = ImageStats {
            width,
            height,
            total_pixels: pixels.len() as u64,
            ..ImageStats::default()
        };

        // Cache exact input colors so repeated pixels take the identical path
        // and repeated runs have a single, order-independent decision point.
        let mut cache: std::collections::HashMap<[u8; 4], (u16, f32, bool)> =
            std::collections::HashMap::new();

        for (i, input) in pixels.iter().enumerate() {
            if input.is_fully_transparent() {
                stats.transparent_input_pixels += 1;
                indices[i] = self.fallback_index as u8;
                continue;
            }
            let (index, dist_sq, fell_back) = cache
                .entry((*input).into())
                .or_insert_with(|| self.decision(FPixel::from_rgba(&self.lut, *input)));
            stats.record(*dist_sq, *dist_sq == 0.0, *fell_back);
            indices[i] = self.final_index(*index, *fell_back) as u8;
        }

        stats.edge_fringe_hits = detect_fringe(pixels, &indices, width, self.fallback_index);
        (indices, stats)
    }

    /// Decide nearest entry and whether the threshold rejected it.
    fn decision(&self, point: FPixel) -> (u16, f32, bool) {
        let (index, dist_sq) = self
            .tree
            .search_excluding(point, &self.transparent_entries);
        let fell_back = self
            .threshold_sq
            .is_some_and(|limit| dist_sq > limit);
        (index, dist_sq, fell_back)
    }

    #[inline]
    fn final_index(&self, nearest: u16, fell_back: bool) -> u16 {
        if fell_back {
            self.fallback_index
        } else {
            nearest
        }
    }

    /// Expand palette indices into literal palette RGBA values. The result can
    /// only contain colors present in the external palette.
    pub fn compose(&self, indices: &[u8]) -> Vec<Rgba8> {
        indices
            .iter()
            .map(|&i| self.palette[usize::from(i)])
            .collect()
    }

    /// Nearest-neighbor search exposed for tests/tooling.
    pub fn nearest(&self, color: Rgba8) -> (u16, f64) {
        let point = FPixel::from_rgba(&self.lut, color);
        let (index, dist_sq) = if color.is_fully_transparent() {
            self.tree.search(point)
        } else {
            self.tree
                .search_excluding(point, &self.transparent_entries)
        };
        (index, internal_dist_to_rmse(f64::from(dist_sq)))
    }
}

/// Count opaque input pixels that (a) sit directly next to a transparent
/// input pixel and (b) were themselves sent to the fallback. These render as
/// a visible halo along sprite edges, so a clean run should report zero.
fn detect_fringe(input: &[Rgba8], indices: &[u8], cols: usize, fallback_index: u16) -> u64 {
    if cols == 0 {
        return 0;
    }
    let rows = input.len() / cols;
    let mut hits = 0u64;
    for row in 0..rows {
        for col in 0..cols {
            let i = row * cols + col;
            if input[i].is_fully_transparent()
                || indices[i] != fallback_index as u8
            {
                continue;
            }
            for (dr, dc) in [(0isize, 1isize), (1, 0), (0, -1), (-1, 0)] {
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr < 0 || nc < 0 || nc as usize >= cols || nr as usize >= rows {
                    continue;
                }
                let j = nr as usize * cols + nc as usize;
                if input[j].is_fully_transparent() {
                    hits += 1;
                    break;
                }
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn palette() -> Vec<Rgba8> {
        vec![
            Rgba8::new(0, 0, 0, 0),
            Rgba8::new(255, 0, 0, 255),
            Rgba8::new(0, 255, 0, 255),
            Rgba8::new(0, 0, 255, 255),
            Rgba8::new(255, 255, 255, 255),
        ]
    }

    #[test]
    fn maps_onto_palette_only() {
        let mapper = PaletteMapper::new(palette(), &MapConfig::default()).unwrap();
        let input = [
            Rgba8::new(250, 5, 5, 255),
            Rgba8::new(2, 250, 3, 255),
            Rgba8::new(0, 0, 0, 0),
        ];
        let (indices, _) = mapper.map(&input, 3, 1);
        let output = mapper.compose(&indices);
        assert_eq!(output[0], Rgba8::new(255, 0, 0, 255));
        assert_eq!(output[1], Rgba8::new(0, 255, 0, 255));
        assert!(output[2].is_fully_transparent());
        for color in &output {
            assert!(palette().contains(color));
        }
    }

    #[test]
    fn threshold_triggers_transparent_fallback() {
        let config = MapConfig {
            max_rmse: Some(1.0),
            fallback: Fallback::Transparent,
            ..MapConfig::default()
        };
        let mapper = PaletteMapper::new(palette(), &config).unwrap();
        // Yellow is far from every palette entry.
        let input = [Rgba8::new(255, 255, 0, 255)];
        let (indices, stats) = mapper.map(&input, 1, 1);
        assert_eq!(stats.fallback_pixels, 1);
        assert!(mapper.compose(&indices)[0].is_fully_transparent());
    }

    #[test]
    fn threshold_can_use_named_fallback_color() {
        let config = MapConfig {
            max_rmse: Some(1.0),
            fallback: Fallback::Color(4),
            ..MapConfig::default()
        };
        let mapper = PaletteMapper::new(palette(), &config).unwrap();
        let input = [Rgba8::new(255, 255, 0, 255)];
        let (indices, stats) = mapper.map(&input, 1, 1);
        assert_eq!(stats.fallback_pixels, 1);
        assert_eq!(mapper.compose(&indices)[0], Rgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn exact_palette_colors_have_zero_distance() {
        let mapper = PaletteMapper::new(palette(), &MapConfig::default()).unwrap();
        for color in palette().iter().filter(|c| !c.is_fully_transparent()) {
            let (_, rmse) = mapper.nearest(*color);
            assert_eq!(rmse, 0.0);
        }
    }

    #[test]
    fn repeated_runs_are_identical() {
        let mapper = PaletteMapper::new(palette(), &MapConfig {
            max_rmse: Some(40.0),
            fallback: Fallback::Transparent,
            ..MapConfig::default()
        })
        .unwrap();
        let input: Vec<Rgba8> = (0..64)
            .map(|i| Rgba8::new((i * 13) as u8, (i * 29) as u8, (i * 7) as u8, 255))
            .collect();
        let (first, _) = mapper.map(&input, 8, 8);
        for _ in 0..5 {
            let (again, _) = mapper.map(&input, 8, 8);
            assert_eq!(first, again);
        }
    }

    #[test]
    fn transparent_fallback_requires_transparent_entry() {
        let opaque: Vec<Rgba8> = palette()
            .into_iter()
            .filter(|c| !c.is_fully_transparent())
            .collect();
        let result = PaletteMapper::new(opaque, &MapConfig::default());
        assert!(result.is_err());
    }
}
