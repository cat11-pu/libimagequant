//! 500-image regression harness for libimagequant.
//!
//! Generates a fixed-seed corpus of 500 RGBA images across six classes
//! (solid / gradient / hardedge / photo / screenshot / transparent), runs the
//! full quantization pipeline twice per image, records quality metrics and
//! timing, verifies byte-level determinism, and exports the worst samples as
//! PNG pairs.
//!
//! Output directory is controlled by `REG_OUT_DIR` (default: `reg-out`).

use imagequant::RGBA;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const N_IMAGES: usize = 500;
const N_WORST: usize = 12;
const SEED: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64 - deterministic across platforms and Rust versions.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0, "rng.below(0)");
        self.next_u32() % n.max(1)
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        if hi <= lo {
            return lo;
        }
        lo + self.below(hi - lo + 1)
    }

    fn pixel(&mut self, alpha: u8) -> RGBA {
        RGBA::new(
            (self.next_u64() & 0xFF) as u8,
            ((self.next_u64() >> 8) & 0xFF) as u8,
            ((self.next_u64() >> 16) & 0xFF) as u8,
            alpha,
        )
    }
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn blend(dst: RGBA, src: RGBA) -> RGBA {
    let sa = src.a as i32;
    let inv = 255 - sa;
    RGBA::new(
        clamp_u8((src.r as i32 * sa + dst.r as i32 * inv) / 255),
        clamp_u8((src.g as i32 * sa + dst.g as i32 * inv) / 255),
        clamp_u8((src.b as i32 * sa + dst.b as i32 * inv) / 255),
        clamp_u8((src.a as i32 * sa + dst.a as i32 * inv) / 255),
    )
}

fn gen_solid(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    let color = rng.pixel(255);
    vec![color; w * h]
}

fn gen_gradient(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    let c00 = rng.pixel(255);
    let c10 = rng.pixel(255);
    let c01 = rng.pixel(255);
    let c11 = rng.pixel(255);
    let band = 1u32 << rng.range(3, 6);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let fy = y as f64 / (h - 1).max(1) as f64;
        for x in 0..w {
            let fx = x as f64 / (w - 1).max(1) as f64;
            let top = |a: u8, b: u8| a as f64 * (1.0 - fx) + b as f64 * fx;
            let bot = |a: u8, b: u8| a as f64 * (1.0 - fx) + b as f64 * fx;
            let r = top(c00.r, c10.r) * (1.0 - fy) + bot(c01.r, c11.r) * fy;
            let g = top(c00.g, c10.g) * (1.0 - fy) + bot(c01.g, c11.g) * fy;
            let b = top(c00.b, c10.b) * (1.0 - fy) + bot(c01.b, c11.b) * fy;
            let step = |v: f64| ((v / band as f64).round() * band as f64) as i32;
            out.push(RGBA::new(
                clamp_u8(step(r)),
                clamp_u8(step(g)),
                clamp_u8(step(b)),
                255,
            ));
        }
    }
    out
}

fn fill_rect(
    buf: &mut [RGBA],
    w: usize,
    h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: RGBA,
) {
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            buf[y * w + x] = color;
        }
    }
}

fn gen_hardedge(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    let mut buf = vec![rng.pixel(255); w * h];
    let n_rects = rng.range(40, 90);
    for _ in 0..n_rects {
        let rw = rng.range(4, (w as u32 / 3).max(5));
        let rh = rng.range(4, (h as u32 / 3).max(5));
        let x0 = rng.below(w as u32) as usize;
        let y0 = rng.below(h as u32) as usize;
        fill_rect(
            &mut buf,
            w,
            h,
            x0,
            y0,
            x0 + rw as usize,
            y0 + rh as usize,
            rng.pixel(255),
        );
    }
    for _ in 0..20 {
        let color = rng.pixel(255);
        if rng.below(2) == 0 {
            let y = rng.below(h as u32) as usize;
            let thickness = rng.range(1, 2) as usize;
            fill_rect(&mut buf, w, h, 0, y, w, (y + thickness).min(h), color);
        } else {
            let x = rng.below(w as u32) as usize;
            let thickness = rng.range(1, 2) as usize;
            fill_rect(&mut buf, w, h, x, 0, (x + thickness).min(w), h, color);
        }
    }
    buf
}

fn gen_photo(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    // Smooth low-frequency base via a small grid of random control colors.
    let gw = rng.range(4, 8) as usize;
    let gh = rng.range(4, 8) as usize;
    let grid: Vec<RGBA> = (0..gw * gh).map(|_| rng.pixel(255)).collect();
    let mut buf = Vec::with_capacity(w * h);
    for y in 0..h {
        let gy = y as f64 / h as f64 * (gh - 1) as f64;
        let y0 = gy.floor() as usize;
        let y1 = (y0 + 1).min(gh - 1);
        let ty = gy - y0 as f64;
        for x in 0..w {
            let gx = x as f64 / w as f64 * (gw - 1) as f64;
            let x0 = gx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let tx = gx - x0 as f64;
            let p00 = grid[y0 * gw + x0];
            let p10 = grid[y0 * gw + x1];
            let p01 = grid[y1 * gw + x0];
            let p11 = grid[y1 * gw + x1];
            let lerp = |a: u8, b: u8, t: f64| a as f64 * (1.0 - t) + b as f64 * t;
            let top_r = lerp(p00.r, p10.r, tx);
            let top_g = lerp(p00.g, p10.g, tx);
            let top_b = lerp(p00.b, p10.b, tx);
            let bot_r = lerp(p01.r, p11.r, tx);
            let bot_g = lerp(p01.g, p11.g, tx);
            let bot_b = lerp(p01.b, p11.b, tx);
            let lerpf = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
            let r = lerpf(top_r, bot_r, ty);
            let g = lerpf(top_g, bot_g, ty);
            let b = lerpf(top_b, bot_b, ty);
            buf.push(RGBA::new(
                clamp_u8(r as i32),
                clamp_u8(g as i32),
                clamp_u8(b as i32),
                255,
            ));
        }
    }
    // Fine-grained photographic noise.
    for px in buf.iter_mut() {
        let n = ((rng.next_u64() & 0x1F) as i32) - 16;
        px.r = clamp_u8(px.r as i32 + n);
        px.g = clamp_u8(px.g as i32 + n);
        px.b = clamp_u8(px.b as i32 + n);
    }
    // Soft overlapping translucent blobs.
    for _ in 0..rng.range(30, 60) {
        let cx = rng.range(0, (w as u32).saturating_sub(1).max(1)) as usize;
        let cy = rng.range(0, (h as u32).saturating_sub(1).max(1)) as usize;
        let rad = rng.range(6, 30) as i32;
        let color = RGBA::new(
            (rng.next_u64() & 0xFF) as u8,
            ((rng.next_u64() >> 8) & 0xFF) as u8,
            ((rng.next_u64() >> 16) & 0xFF) as u8,
            rng.range(20, 90) as u8,
        );
        let ylo = cy.saturating_sub(rad as usize);
        let yhi = (cy + rad as usize).min(h);
        let xlo = cx.saturating_sub(rad as usize);
        let xhi = (cx + rad as usize).min(w);
        for y in ylo..yhi {
            for x in xlo..xhi {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                if dx * dx + dy * dy <= rad * rad {
                    let idx = y * w + x;
                    buf[idx] = blend(buf[idx], color);
                }
            }
        }
    }
    buf
}

fn gen_screenshot(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    let bg = RGBA::new(240, 240, 243, 255);
    let mut buf = vec![bg; w * h];
    let bar_h = rng.range(12, 20) as usize;
    fill_rect(&mut buf, w, h, 0, 0, w, bar_h, RGBA::new(48, 52, 64, 255));
    for i in 0..3u32 {
        let bx = 6 + i * 14;
        fill_rect(
            &mut buf,
            w,
            h,
            bx as usize,
            bar_h / 2 - 3,
            (bx + 6) as usize,
            bar_h / 2 + 3,
            RGBA::new((220 - i * 30) as u8, (90 + i * 40) as u8, 80, 255),
        );
    }
    // Flat UI panels with crisp 1px borders.
    let n_panels = rng.range(8, 16);
    for _ in 0..n_panels {
        let pw = rng.range(20, (w as u32 / 3).max(20)).min(w as u32 - 1) as usize;
        let ph = rng.range(16, (h as u32 / 3).max(16)).min(h as u32 - 1) as usize;
        let x0 = rng.below((w - pw) as u32) as usize;
        let y_hi = (h - ph) as u32;
        let y0 = rng.range(bar_h as u32 + 2, y_hi.max(bar_h as u32 + 2)) as usize;
        let fill = RGBA::new(
            225 + rng.below(25) as u8,
            225 + rng.below(25) as u8,
            228 + rng.below(22) as u8,
            255,
        );
        fill_rect(&mut buf, w, h, x0, y0, x0 + pw, y0 + ph, fill);
        let border = RGBA::new(120, 124, 132, 255);
        fill_rect(&mut buf, w, h, x0, y0, x0 + pw, y0 + 1, border);
        fill_rect(&mut buf, w, h, x0, y0 + ph - 1, x0 + pw, y0 + ph, border);
        fill_rect(&mut buf, w, h, x0, y0, x0 + 1, y0 + ph, border);
        fill_rect(&mut buf, w, h, x0 + pw - 1, y0, x0 + pw, y0 + ph, border);
    }
    // Text-like horizontal runs of dark pixels.
    for _ in 0..rng.range(20, 40) {
        let color = RGBA::new(
            60 + rng.below(40) as u8,
            60 + rng.below(40) as u8,
            64 + rng.below(40) as u8,
            255,
        );
        let run_w = rng.range(8, 80).min(w as u32 - 1) as usize;
        let run_h = rng.range(1, 3).min(h as u32 - 1) as usize;
        let x0 = rng.below((w - run_w) as u32) as usize;
        let y_hi = (h - run_h) as u32;
        let y0 = rng.range(bar_h as u32 + 4, y_hi.max(bar_h as u32 + 4)) as usize;
        fill_rect(&mut buf, w, h, x0, y0, x0 + run_w, y0 + run_h, color);
    }
    buf
}

fn gen_transparent(rng: &mut Rng, w: usize, h: usize) -> Vec<RGBA> {
    // Start fully transparent, then paint semi-transparent shapes so alpha
    // values span the whole range and edges are partially covered.
    let mut buf = vec![RGBA::new(0, 0, 0, 0); w * h];
    for _ in 0..rng.range(20, 45) {
        let shape = rng.below(3);
        let color = RGBA::new(
            (rng.next_u64() & 0xFF) as u8,
            ((rng.next_u64() >> 8) & 0xFF) as u8,
            ((rng.next_u64() >> 16) & 0xFF) as u8,
            rng.range(20, 255) as u8,
        );
        if shape == 0 {
            let rw = rng.range(8, w as u32 / 2) as usize;
            let rh = rng.range(8, h as u32 / 2) as usize;
            let x0 = rng.below((w - rw + 1) as u32) as usize;
            let y0 = rng.below((h - rh + 1) as u32) as usize;
            fill_rect(&mut buf, w, h, x0, y0, x0 + rw, y0 + rh, color);
        } else {
            let cx = rng.range(0, (w as u32).saturating_sub(1).max(1)) as usize;
            let cy = rng.range(0, (h as u32).saturating_sub(1).max(1)) as usize;
            let rad = rng.range(5, 40) as i32;
            let ylo = cy.saturating_sub(rad as usize);
            let yhi = (cy + rad as usize).min(h);
            let xlo = cx.saturating_sub(rad as usize);
            let xhi = (cx + rad as usize).min(w);
            for y in ylo..yhi {
                for x in xlo..xhi {
                    let dx = x as i32 - cx as i32;
                    let dy = y as i32 - cy as i32;
                    if dx * dx + dy * dy <= rad * rad {
                        let idx = y * w + x;
                        buf[idx] = blend(buf[idx], color);
                    }
                }
            }
        }
    }
    // Feathered alpha on a few edges: fade columns/rows near shape borders.
    for _ in 0..rng.range(2, 6) {
        if rng.below(2) == 0 {
            let x = rng.below(w as u32) as usize;
            let width = rng.range(2, 8) as usize;
            for y in 0..h {
                for (step, xx) in (x..(x + width).min(w)).enumerate() {
                    let idx = y * w + xx;
                    let fade = (255 * (width - step) / width.max(1)) as i32;
                    buf[idx].a = clamp_u8((buf[idx].a as i32 * fade) / 255);
                }
            }
        } else {
            let y = rng.below(h as u32) as usize;
            let height = rng.range(2, 8) as usize;
            for (step, yy) in (y..(y + height).min(h)).enumerate() {
                for x in 0..w {
                    let idx = yy * w + x;
                    let fade = (255 * (height - step) / height.max(1)) as i32;
                    buf[idx].a = clamp_u8((buf[idx].a as i32 * fade) / 255);
                }
            }
        }
    }
    buf
}

struct Sample {
    id: usize,
    class: &'static str,
    width: usize,
    height: usize,
    pixels: Vec<RGBA>,
}

fn generate_corpus() -> Vec<Sample> {
    let classes: [(&str, fn(&mut Rng, usize, usize) -> Vec<RGBA>); 6] = [
        ("solid", gen_solid),
        ("gradient", gen_gradient),
        ("hardedge", gen_hardedge),
        ("photo", gen_photo),
        ("screenshot", gen_screenshot),
        ("transparent", gen_transparent),
    ];
    // Class boundaries: five classes of 83, transparent gets 85 (ids 415..=499).
    let mut samples = Vec::with_capacity(N_IMAGES);
    for id in 0..N_IMAGES {
        let class_idx = (id / 83).min(classes.len() - 1);
        let (class, generator) = classes[class_idx];
        let mut rng = Rng::new(SEED ^ (id as u64).wrapping_mul(0xD1B54A32D192ED03));
        let width = rng.range(64, 320) as usize;
        let height = rng.range(64, 320) as usize;
        let pixels = generator(&mut rng, width, height);
        assert_eq!(pixels.len(), width * height);
        samples.push(Sample {
            id,
            class,
            width,
            height,
            pixels,
        });
    }
    samples
}

struct QuantOutput {
    palette: Vec<RGBA>,
    indices: Vec<u8>,
    quant_quality: u8,
    quant_error: f64,
    remap_error: f64,
}

fn run_pipeline(pixels: &[RGBA], width: usize, height: usize) -> (QuantOutput, u128) {
    let start = Instant::now();
    let mut attr = imagequant::new();
    attr.set_speed(5).unwrap();
    attr.set_quality(0, 90).unwrap();
    attr.set_max_colors(256).unwrap();
    let mut image = attr.new_image_borrowed(pixels, width, height, 0.0).unwrap();
    let mut result = attr.quantize(&mut image).unwrap();
    result.set_dithering_level(1.0).unwrap();
    let (palette, indices) = result.remapped(&mut image).unwrap();
    let elapsed = start.elapsed().as_nanos();
    let out = QuantOutput {
        palette,
        indices,
        quant_quality: result.quantization_quality().unwrap(),
        quant_error: result.quantization_error().unwrap(),
        remap_error: result.remapping_error().unwrap(),
    };
    (out, elapsed)
}

fn reconstruct(out: &QuantOutput) -> Vec<RGBA> {
    out.indices
        .iter()
        .map(|&idx| out.palette[idx as usize])
        .collect()
}

fn mean_squared_error(original: &[RGBA], quantized: &[RGBA]) -> f64 {
    // Mirrors the library semantics: the RGB of fully-transparent pixels is
    // unobservable, so only the alpha channel counts when both sides are
    // fully transparent. All other pixels compare all four channels.
    let mut sum = 0u128;
    let mut channels = 0u128;
    for (a, b) in original.iter().zip(quantized.iter()) {
        let da = a.a as i32 - b.a as i32;
        sum += (da * da) as u128;
        channels += 1;
        if a.a != 0 || b.a != 0 {
            let dr = a.r as i32 - b.r as i32;
            let dg = a.g as i32 - b.g as i32;
            let db = a.b as i32 - b.b as i32;
            sum += (dr * dr + dg * dg + db * db) as u128;
            channels += 3;
        }
    }
    (sum as f64) / (channels as f64)
}

struct MetricRow {
    id: usize,
    class: &'static str,
    width: usize,
    height: usize,
    pixels: usize,
    palette_len: usize,
    quant_quality: u8,
    quant_error: f64,
    remap_error: f64,
    mse: f64,
    psnr: f64,
    deterministic: bool,
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("REG_OUT_DIR").unwrap_or_else(|_| "reg-out".into()));
    let worst_dir = out_dir.join("worst");
    fs::create_dir_all(&worst_dir).unwrap();

    let corpus = generate_corpus();
    let mut metrics = Vec::with_capacity(N_IMAGES);
    let timing_path = out_dir.join("timing.jsonl");
    let metrics_path = out_dir.join("metrics.jsonl");
    let timing_file = BufWriter::new(fs::File::create(&timing_path).unwrap());
    let metrics_file = BufWriter::new(fs::File::create(&metrics_path).unwrap());
    let mut timing_w = timing_file;
    let mut metrics_w = metrics_file;

    for sample in &corpus {
        let (first, time_ns) = run_pipeline(&sample.pixels, sample.width, sample.height);
        let (second, _) = run_pipeline(&sample.pixels, sample.width, sample.height);
        let deterministic = first.palette == second.palette && first.indices == second.indices;

        let quantized = reconstruct(&first);
        let mse = mean_squared_error(&sample.pixels, &quantized);
        let psnr = if mse <= 0.0 {
            999.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };

        let row = MetricRow {
            id: sample.id,
            class: sample.class,
            width: sample.width,
            height: sample.height,
            pixels: sample.pixels.len(),
            palette_len: first.palette.len(),
            quant_quality: first.quant_quality,
            quant_error: first.quant_error,
            remap_error: first.remap_error,
            mse,
            psnr,
            deterministic,
        };
        writeln!(timing_w, "{{\"id\":{},\"time_ns\":{}}}", row.id, time_ns).unwrap();
        writeln!(
            metrics_w,
            "{{\"id\":{},\"class\":\"{}\",\"width\":{},\"height\":{},\"pixels\":{},\"palette_len\":{},\"quant_quality\":{},\"quant_error\":{:.8},\"remap_error\":{:.8},\"mse\":{:.8},\"psnr\":{:.6},\"deterministic\":{}}}",
            row.id,
            row.class,
            row.width,
            row.height,
            row.pixels,
            row.palette_len,
            row.quant_quality,
            row.quant_error,
            row.remap_error,
            row.mse,
            row.psnr,
            row.deterministic
        )
        .unwrap();
        metrics.push(row);
    }
    timing_w.flush().unwrap();
    metrics_w.flush().unwrap();

    // Export worst N samples as original/quantized PNG pairs.
    let mut order: Vec<usize> = (0..N_IMAGES).collect();
    order.sort_by(|&a, &b| {
        metrics[b]
            .mse
            .partial_cmp(&metrics[a].mse)
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    for &idx in order.iter().take(N_WORST) {
        let sample = &corpus[idx];
        let (out, _) = run_pipeline(&sample.pixels, sample.width, sample.height);
        let quantized = reconstruct(&out);
        let stem = format!(
            "{}_{}_{}x{}",
            sample.id, sample.class, sample.width, sample.height
        );
        lodepng::encode32_file(
            worst_dir.join(format!("{stem}.orig.png")),
            &sample.pixels,
            sample.width,
            sample.height,
        )
        .unwrap();
        lodepng::encode32_file(
            worst_dir.join(format!("{stem}.quant.png")),
            &quantized,
            sample.width,
            sample.height,
        )
        .unwrap();
    }

    let nondet = metrics.iter().filter(|r| !r.deterministic).count();
    eprintln!(
        "done: {} images, nondeterministic={}, out={}",
        N_IMAGES,
        nondet,
        out_dir.display()
    );
}
