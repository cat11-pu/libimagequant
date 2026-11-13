//! Batch directory processing, aggregated statistics and JSON reporting.

use crate::color::Rgba8;
use crate::mapper::{ImageStats, PaletteMapper};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct InputImage {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Rgba8>,
}

#[derive(Clone, Debug)]
pub struct ImageReport {
    pub relative_path: String,
    pub stats: ImageStats,
    pub mean_rmse: f64,
    pub p95_rmse: f64,
    pub p99_rmse: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BatchReport {
    pub image_count: usize,
    pub total_pixels: u64,
    pub total_mapped: u64,
    pub total_fallback: u64,
    pub total_transparent_input: u64,
    pub total_exact: u64,
    pub total_edge_fringe: u64,
    pub mean_rmse_weighted: f64,
    pub worst_mean_rmse: f64,
    pub worst_max_rmse: f64,
    pub images: Vec<ImageReport>,
}

pub fn collect_pngs(input_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    walk(input_dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read directory {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        if file_type.is_dir() {
            walk(&path, out)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"))
        {
            out.push(path);
        }
    }
    Ok(())
}

pub fn load_image(path: &Path) -> Result<InputImage, String> {
    let bitmap =
        lodepng::decode32_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let pixels = bitmap.buffer.iter().map(|c| Rgba8::new(c.r, c.g, c.b, c.a)).collect();
    Ok(InputImage {
        path: path.to_path_buf(),
        width: bitmap.width,
        height: bitmap.height,
        pixels,
    })
}

pub fn save_rgba_png(path: &Path, pixels: &[Rgba8], width: usize, height: usize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let bytes: Vec<[u8; 4]> = pixels.iter().map(|c| (*c).into()).collect();
    lodepng::encode32_file(path, &bytes, width, height).map_err(|e| e.to_string())
}

/// Map every image with the shared mapper and build the aggregate report.
pub fn run_batch(
    mapper: &PaletteMapper,
    images: &[InputImage],
    root: &Path,
) -> BatchReport {
    let mut report = BatchReport::default();
    for image in images {
        let (indices, stats) = mapper.map(&image.pixels, image.width, image.height);
        let composed = mapper.compose(&indices);

        // Defense-in-depth: output must only ever contain palette colors.
        debug_assert!(composed.iter().all(|c| mapper.palette().contains(c)));

        let relative = image
            .path
            .strip_prefix(root)
            .unwrap_or(&image.path)
            .display()
            .to_string();
        let mean = stats.mean_rmse();
        let image_report = ImageReport {
            relative_path: relative,
            mean_rmse: mean,
            p95_rmse: stats.percentile_rmse(95.0),
            p99_rmse: stats.percentile_rmse(99.0),
            stats,
        };
        report.total_pixels += image_report.stats.total_pixels;
        report.total_mapped += image_report.stats.mapped_pixels();
        report.total_fallback += image_report.stats.fallback_pixels;
        report.total_transparent_input += image_report.stats.transparent_input_pixels;
        report.total_exact += image_report.stats.exact_matches;
        report.total_edge_fringe += image_report.stats.edge_fringe_hits;
        report.mean_rmse_weighted += mean * image_report.stats.mapped_pixels() as f64;
        report.worst_mean_rmse = report.worst_mean_rmse.max(mean);
        report.worst_max_rmse = report.worst_max_rmse.max(image_report.stats.max_dist_rmse);
        report.images.push(image_report);
    }

    report.image_count = report.images.len();
    if report.total_mapped > 0 {
        report.mean_rmse_weighted /= report.total_mapped as f64;
    }
    report
}

/// Worst samples ordered by mean RMSE, then max RMSE, then path for stable ties.
pub fn worst_samples(report: &BatchReport, limit: usize) -> Vec<&ImageReport> {
    let mut sorted: Vec<&ImageReport> = report.images.iter().collect();
    sorted.sort_by(|a, b| {
        b.mean_rmse
            .total_cmp(&a.mean_rmse)
            .then(b.stats.max_dist_rmse.total_cmp(&a.stats.max_dist_rmse))
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    sorted.truncate(limit);
    sorted
}

fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 2);
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn f64_4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// Write the full deterministic JSON report.
pub fn write_json_report(
    writer: &mut impl Write,
    report: &BatchReport,
    worst: &[&ImageReport],
    palette_size: usize,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    writeln!(writer, "  \"summary\": {{")?;
    writeln!(writer, "    \"palette_size\": {},", palette_size)?;
    writeln!(writer, "    \"image_count\": {},", report.image_count)?;
    writeln!(writer, "    \"total_pixels\": {},", report.total_pixels)?;
    writeln!(writer, "    \"mapped_pixels\": {},", report.total_mapped)?;
    writeln!(writer, "    \"exact_match_pixels\": {},", report.total_exact)?;
    writeln!(writer, "    \"fallback_pixels\": {},", report.total_fallback)?;
    writeln!(writer, "    \"transparent_input_pixels\": {},", report.total_transparent_input)?;
    writeln!(writer, "    \"edge_fringe_hits\": {},", report.total_edge_fringe)?;
    writeln!(writer, "    \"mean_rmse\": {},", f64_4(report.mean_rmse_weighted))?;
    writeln!(writer, "    \"worst_image_mean_rmse\": {},", f64_4(report.worst_mean_rmse))?;
    writeln!(writer, "    \"worst_pixel_rmse\": {}", f64_4(report.worst_max_rmse))?;
    writeln!(writer, "  }},")?;

    writeln!(writer, "  \"worst_samples\": [")?;
    for (i, sample) in worst.iter().enumerate() {
        let comma = if i + 1 == worst.len() { "" } else { "," };
        writeln!(
            writer,
            "    {{\"path\": \"{}\", \"mean_rmse\": {}, \"p95_rmse\": {}, \"p99_rmse\": {}, \"max_rmse\": {}, \"fallback_pixels\": {}, \"edge_fringe_hits\": {}}}{}",
            json_escape(&sample.relative_path),
            f64_4(sample.mean_rmse),
            f64_4(sample.p95_rmse),
            f64_4(sample.p99_rmse),
            f64_4(sample.stats.max_dist_rmse),
            sample.stats.fallback_pixels,
            sample.stats.edge_fringe_hits,
            comma
        )?;
    }
    writeln!(writer, "  ],")?;

    writeln!(writer, "  \"images\": [")?;
    for (i, image) in report.images.iter().enumerate() {
        let comma = if i + 1 == report.images.len() { "" } else { "," };
        writeln!(
            writer,
            "    {{\"path\": \"{}\", \"width\": {}, \"height\": {}, \"total_pixels\": {}, \"mapped_pixels\": {}, \"exact_matches\": {}, \"fallback_pixels\": {}, \"transparent_input_pixels\": {}, \"edge_fringe_hits\": {}, \"mean_rmse\": {}, \"p95_rmse\": {}, \"p99_rmse\": {}, \"max_rmse\": {}}}{}",
            json_escape(&image.relative_path),
            image.stats.width,
            image.stats.height,
            image.stats.total_pixels,
            image.stats.mapped_pixels(),
            image.stats.exact_matches,
            image.stats.fallback_pixels,
            image.stats.transparent_input_pixels,
            image.stats.edge_fringe_hits,
            f64_4(image.mean_rmse),
            f64_4(image.p95_rmse),
            f64_4(image.p99_rmse),
            f64_4(image.stats.max_dist_rmse),
            comma
        )?;
    }
    writeln!(writer, "  ]")?;
    writeln!(writer, "}}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_quotes_and_backslashes() {
        assert_eq!(json_escape("a\"b\\c"), "a\\\"b\\\\c");
    }

    #[test]
    fn worst_samples_are_stable() {
        let make = |path: &str, mean: f64| ImageReport {
            relative_path: path.to_string(),
            stats: ImageStats {
                max_dist_rmse: mean,
                ..ImageStats::default()
            },
            mean_rmse: mean,
            p95_rmse: 0.0,
            p99_rmse: 0.0,
        };
        let report = BatchReport {
            images: vec![make("a.png", 1.0), make("b.png", 2.0), make("c.png", 2.0)],
            ..BatchReport::default()
        };
        let worst = worst_samples(&report, 2);
        assert_eq!(worst[0].relative_path, "b.png");
        assert_eq!(worst[1].relative_path, "c.png");
    }
}
