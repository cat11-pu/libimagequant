//! External palette loading from PNG swatches or plain-text hex lists.

use crate::color::Rgba8;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug)]
pub enum PaletteError {
    Io(String),
    Png(String),
    BadEntry { line: usize, text: String },
    TooMany(usize),
}

impl std::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PaletteError::Io(e) => write!(f, "cannot read palette file: {e}"),
            PaletteError::Png(e) => write!(f, "cannot decode palette PNG: {e}"),
            PaletteError::BadEntry { line, text } => {
                write!(f, "bad palette entry on line {line}: {text:?}")
            }
            PaletteError::TooMany(n) => write!(f, "palette has {n} colors, at most 256 allowed"),
        }
    }
}

impl std::error::Error for PaletteError {}

pub const MAX_PALETTE_SIZE: usize = 256;

/// Load a palette, deduping colors while preserving first-seen order.
///
/// Supported formats:
/// * `.png` — every distinct pixel color, in stable raster order
///   (left-to-right, top-to-bottom), becomes a palette entry.
/// * anything else — UTF-8 text with one `#RGB`, `#RGBA`, `#RRGGBB` or
///   `#RRGGBBAA` entry per line; `#` comments and blank lines are ignored.
pub fn load_palette(path: &Path) -> Result<Vec<Rgba8>, PaletteError> {
    let is_png = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    let colors = if is_png {
        load_png_palette(path)?
    } else {
        load_text_palette(path)?
    };

    let mut seen: BTreeSet<[u8; 4]> = BTreeSet::new();
    let mut unique = Vec::new();
    for color in colors {
        if seen.insert(color.into()) {
            unique.push(color);
        }
    }
    if unique.len() > MAX_PALETTE_SIZE {
        return Err(PaletteError::TooMany(unique.len()));
    }
    if unique.is_empty() {
        return Err(PaletteError::BadEntry {
            line: 0,
            text: "palette contains no colors".to_string(),
        });
    }
    Ok(unique)
}

fn load_png_palette(path: &Path) -> Result<Vec<Rgba8>, PaletteError> {
    let bitmap =
        lodepng::decode32_file(path).map_err(|e| PaletteError::Png(e.to_string()))?;
    Ok(bitmap.buffer.iter().map(Rgba8::from_bytes).collect())
}

fn load_text_palette(path: &Path) -> Result<Vec<Rgba8>, PaletteError> {
    let text = std::fs::read_to_string(path).map_err(|e| PaletteError::Io(e.to_string()))?;
    let mut colors = Vec::new();
    for (line_no, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let token = line.split_whitespace().next().unwrap_or(line);
        // A '#' followed by a space (or end of line) is a comment marker;
        // '#rgb...' without whitespace is a hex color literal.
        if token == "#" {
            continue;
        }
        let color = parse_hex_color(token).ok_or_else(|| PaletteError::BadEntry {
            line: line_no + 1,
            text: raw_line.to_string(),
        })?;
        colors.push(color);
    }
    Ok(colors)
}

fn parse_hex_color(token: &str) -> Option<Rgba8> {
    let hex = token.strip_prefix('#').unwrap_or(token);
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let nibble = |n: usize| -> Option<u8> {
        hex.as_bytes()
            .get(n)
            .and_then(|b| (*b as char).to_digit(16))
            .map(|d| d as u8)
    };
    let pair = |n: usize| -> Option<u8> { Some(nibble(n)? << 4 | nibble(n + 1)?) };
    match hex.len() {
        3 => {
            let r = nibble(0)?;
            let g = nibble(1)?;
            let b = nibble(2)?;
            Some(Rgba8::new(r * 17, g * 17, b * 17, 255))
        }
        4 => {
            let r = nibble(0)?;
            let g = nibble(1)?;
            let b = nibble(2)?;
            let a = nibble(3)?;
            Some(Rgba8::new(r * 17, g * 17, b * 17, a * 17))
        }
        6 => Some(Rgba8::new(pair(0)?, pair(2)?, pair(4)?, 255)),
        8 => Some(Rgba8::new(pair(0)?, pair(2)?, pair(4)?, pair(6)?)),
        _ => None,
    }
}

impl Rgba8 {
    fn from_bytes(c: &lodepng::RGBA) -> Self {
        Self::new(c.r, c.g, c.b, c.a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_hex_shapes() {
        assert_eq!(
            parse_hex_color("#abc").unwrap(),
            Rgba8::new(0xaa, 0xbb, 0xcc, 255)
        );
        assert_eq!(
            parse_hex_color("#abcd").unwrap(),
            Rgba8::new(0xaa, 0xbb, 0xcc, 0xdd)
        );
        assert_eq!(
            parse_hex_color("112233").unwrap(),
            Rgba8::new(0x11, 0x22, 0x33, 255)
        );
        assert_eq!(
            parse_hex_color("#11223344").unwrap(),
            Rgba8::new(0x11, 0x22, 0x33, 0x44)
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_hex_color("#12").is_none());
        assert!(parse_hex_color("#xyzxyz").is_none());
        assert!(parse_hex_color("12345").is_none());
    }

    #[test]
    fn text_loader_distinguishes_comments_from_colors() {
        let dir = std::env::temp_dir().join("palette_map_text_palette");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pal.txt");
        std::fs::write(
            &path,
            "# this is a comment line\n#ff0000\n\n#\n#00ff00cc\n",
        )
        .unwrap();
        let colors = load_text_palette(&path).unwrap();
        assert_eq!(
            colors,
            vec![
                Rgba8::new(255, 0, 0, 255),
                Rgba8::new(0, 255, 0, 204),
            ]
        );
    }
}
