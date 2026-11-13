//! Deterministic k-d tree nearest-neighbor search over palette points.
//!
//! Ties on distance are broken by the smallest palette index, so the result is
//! bit-for-bit identical to an exhaustive brute-force search, and therefore
//! identical across runs and platforms (all math is plain `f32` with a fixed
//! evaluation order).

use crate::color::FPixel;

#[derive(Clone, Copy)]
struct AxisPoint {
    point: FPixel,
    index: u16,
}

enum Node {
    Empty,
    Leaf {
        point: FPixel,
        index: u16,
        axis: u8,
        left: Box<Node>,
        right: Box<Node>,
    },
}

pub struct NearestTree {
    root: Node,
    palette: Vec<FPixel>,
}

impl NearestTree {
    /// Build a tree for the given palette points.
    ///
    /// The palette order defines tie-breaking: when two entries are exactly
    /// equidistant, the entry with the lower index wins.
    pub fn build(palette: Vec<FPixel>) -> Result<Self, &'static str> {
        if palette.is_empty() {
            return Err("palette must contain at least one color");
        }
        if palette.len() > u16::MAX as usize + 1 {
            return Err("palette must contain at most 65536 colors");
        }
        let mut items: Vec<AxisPoint> = palette
            .iter()
            .enumerate()
            .map(|(index, &point)| AxisPoint {
                point,
                index: index as u16,
            })
            .collect();
        let root = build_node(&mut items, 0);
        Ok(Self { root, palette })
    }

 #[inline]
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }

    /// Find the closest palette entry. Returns `(palette_index, squared_distance)`.
    #[inline]
    pub fn search(&self, needle: FPixel) -> (u16, f32) {
        self.search_excluding(needle, &[])
    }

    /// Find the closest palette entry, ignoring every palette index in
    /// `excluded`. Returns `(palette_index, squared_distance)`.
    #[inline]
    pub fn search_excluding(&self, needle: FPixel, excluded: &[u16]) -> (u16, f32) {
        let mut best_index = 0u16;
        let mut best_dist = f32::INFINITY;
        search_node(&self.root, needle, excluded, &mut best_index, &mut best_dist);
        (best_index, best_dist)
    }
}

#[inline]
fn axis_value(point: FPixel, axis: u8) -> f32 {
    match axis % 4 {
        0 => point.a,
        1 => point.r,
        2 => point.g,
        _ => point.b,
    }
}

fn build_node(items: &mut [AxisPoint], depth: u8) -> Node {
    if items.is_empty() {
        return Node::Empty;
    }
    let axis = depth % 4;
    // Deterministic ordering even when axis values are equal.
    items.sort_by(|x, y| {
        axis_value(x.point, axis)
            .total_cmp(&axis_value(y.point, axis))
            .then(x.index.cmp(&y.index))
    });
    let median = items.len() / 2;
    let (left, rest) = items.split_at_mut(median);
    let (head, right) = rest.split_first_mut().unwrap();
    let point = head.point;
    let index = head.index;
    Node::Leaf {
        point,
        index,
        axis,
        left: Box::new(build_node(left, depth + 1)),
        right: Box::new(build_node(right, depth + 1)),
    }
}

#[inline]
fn better(dist: f32, index: u16, best_dist: f32, best_index: u16) -> bool {
    dist < best_dist || (dist == best_dist && index < best_index)
}

fn search_node(
    node: &Node,
    needle: FPixel,
    excluded: &[u16],
    best_index: &mut u16,
    best_dist: &mut f32,
) {
    let Node::Leaf {
        point,
        index,
        axis,
        left,
        right,
    } = node
    else {
        return;
    };

    if !excluded.contains(index) {
        let dist = point.diff(needle);
        if better(dist, *index, *best_dist, *best_index) {
            *best_dist = dist;
            *best_index = *index;
        }
    }

    let delta = axis_value(needle, *axis) - axis_value(*point, *axis);
    let (near, far) = if delta <= 0.0 {
        (left.as_ref(), right.as_ref())
    } else {
        (right.as_ref(), left.as_ref())
    };
    search_node(near, needle, excluded, best_index, best_dist);
    if delta * delta <= *best_dist {
        search_node(far, needle, excluded, best_index, best_dist);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{Rgba8, FPixel as FP};

    fn palette(colors: &[Rgba8]) -> (Vec<FPixel>, [f32; 256]) {
        let lut = FP::gamma_lut(0.0);
        (
            colors.iter().map(|c| FP::from_rgba(&lut, *c)).collect(),
            lut,
        )
    }

    fn brute_force(points: &[FPixel], needle: FP) -> (u16, f32) {
        let mut best = (0u16, f32::INFINITY);
        for (i, p) in points.iter().enumerate() {
            let d = p.diff(needle);
            if better(d, i as u16, best.1, best.0) {
                best = (i as u16, d);
            }
        }
        best
    }

    #[test]
    fn matches_brute_force() {
        let colors: Vec<Rgba8> = (0..200)
            .map(|i| Rgba8::new((i * 37) as u8, (i * 91) as u8, (i * 173) as u8, 255))
            .collect();
        let (points, lut) = palette(&colors);
        let tree = NearestTree::build(points.clone()).unwrap();
        for r in 0..16u16 {
            for g in 0..16u16 {
                for b in 0..16u16 {
                    let needle = FP::from_rgba(
                        &lut,
                        Rgba8::new((r * 17) as u8, (g * 17) as u8, (b * 17) as u8, 255),
                    );
                    assert_eq!(tree.search(needle), brute_force(&points, needle));
                }
            }
        }
    }

    #[test]
    fn tie_prefers_lower_index() {
        let colors = [
            Rgba8::new(0, 0, 0, 255),
            Rgba8::new(10, 10, 10, 255),
            Rgba8::new(10, 10, 10, 255),
        ];
        let (points, lut) = palette(&colors);
        let tree = NearestTree::build(points).unwrap();
        let needle = FP::from_rgba(&lut, Rgba8::new(10, 10, 10, 255));
        let (index, dist) = tree.search(needle);
        assert_eq!(dist, 0.0);
        assert_eq!(index, 1);
    }

    #[test]
    fn works_with_alpha_colors() {
        let colors = [
            Rgba8::new(0, 0, 0, 0),
            Rgba8::new(255, 255, 255, 128),
            Rgba8::new(200, 30, 90, 255),
        ];
        let (points, lut) = palette(&colors);
        let tree = NearestTree::build(points.clone()).unwrap();
        for c in colors {
            let needle = FP::from_rgba(&lut, c);
            assert_eq!(tree.search(needle), brute_force(&points, needle));
        }
    }

    #[test]
    fn rejects_empty_palette() {
        assert!(NearestTree::build(Vec::new()).is_err());
    }

    #[test]
    fn exclusion_skips_listed_entries() {
        let colors = [
            Rgba8::new(0, 0, 0, 0),
            Rgba8::new(0, 0, 0, 255),
            Rgba8::new(255, 255, 255, 255),
        ];
        let (points, lut) = palette(&colors);
        let tree = NearestTree::build(points).unwrap();
        let needle = FP::from_rgba(&lut, Rgba8::new(1, 1, 1, 255));
        let (index, _) = tree.search(needle);
        assert_eq!(index, 1);
        let (index_excluded, _) = tree.search_excluding(needle, &[1]);
        assert_eq!(index_excluded, 0);
        let (index_zero, _) = tree.search_excluding(needle, &[0, 1]);
        assert_eq!(index_zero, 2);
    }
}
