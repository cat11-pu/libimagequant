use palette_map::batch::{
    collect_pngs, load_image, run_batch, save_rgba_png, worst_samples, write_json_report,
};
use palette_map::fixture::{generate_fixtures, make_palette_256, write_palette_png};
use palette_map::mapper::{Fallback, MapConfig, PaletteMapper};
use palette_map::palette::load_palette;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

struct Args {
    palette_path: PathBuf,
    input: PathBuf,
    output_dir: PathBuf,
    report_path: PathBuf,
    max_rmse: Option<f64>,
    fallback: Fallback,
    worst_count: usize,
}

fn print_usage(program: &str) {
    eprintln!(
        "\
Usage:
  {program} map --palette <palette.png|palette.txt> --input <dir|file.png> \
--output <dir> --report <report.json> [options]
  {program} make-fixtures --output <dir> [--count 500]

Map options:
  --max-rmse <f64>          reject nearest matches worse than this RMSE (0..255
                            scale, mean of RGB squared errors); default: no limit
  --fallback transparent    rejected pixels become the palette's transparent
                            entry (default)
  --fallback-color <index>  rejected pixels become this palette index
  --worst <n>               number of worst samples listed in the report (default 10)

Palette formats:
  PNG           distinct pixel colors, raster order
  text          one #RGB/#RGBA/#RRGGBB/#RRGGBB color per line; a lone '#'
                token on a line starts a comment
"
    );
}

fn flag_value(
    flag: &str,
    iter: &mut std::slice::Iter<'_, String>,
) -> Result<String, String> {
    iter.next()
        .cloned()
        .ok_or_else(|| format!("flag {flag} requires a value"))
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut palette_path = None;
    let mut input = None;
    let mut output_dir = None;
    let mut report_path = None;
    let mut max_rmse = None;
    let mut fallback = Fallback::Transparent;
    let mut worst_count = 10usize;

    // argv includes the program name and the `map` subcommand.
    let args_only = &argv[2..];
    let mut iter = args_only.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--palette" => palette_path = Some(PathBuf::from(flag_value(flag, &mut iter)?)),
            "--input" => input = Some(PathBuf::from(flag_value(flag, &mut iter)?)),
            "--output" => output_dir = Some(PathBuf::from(flag_value(flag, &mut iter)?)),
            "--report" => report_path = Some(PathBuf::from(flag_value(flag, &mut iter)?)),
            "--max-rmse" => {
                let raw = flag_value(flag, &mut iter)?;
                let parsed: f64 = raw
                    .parse()
                    .map_err(|_| format!("invalid --max-rmse value: {raw}"))?;
                if !(parsed.is_finite() && parsed >= 0.0) {
                    return Err("--max-rmse must be a finite, non-negative number".to_string());
                }
                max_rmse = Some(parsed);
            }
            "--fallback" => {
                let raw = flag_value(flag, &mut iter)?;
                fallback = match raw.as_str() {
                    "transparent" => Fallback::Transparent,
                    other => {
                        return Err(format!("unsupported --fallback mode: {other}"));
                    }
                };
            }
            "--fallback-color" => {
                let raw = flag_value(flag, &mut iter)?;
                let index: u16 = raw
                    .parse()
                    .map_err(|_| format!("invalid --fallback-color index: {raw}"))?;
                fallback = Fallback::Color(index);
            }
            "--worst" => {
                let raw = flag_value(flag, &mut iter)?;
                worst_count = raw
                    .parse()
                    .map_err(|_| format!("invalid --worst count: {raw}"))?;
            }
            "-h" | "--help" => {
                print_usage("palette-map");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        palette_path: palette_path.ok_or("missing --palette")?,
        input: input.ok_or("missing --input")?,
        output_dir: output_dir.ok_or("missing --output")?,
        report_path: report_path.ok_or("missing --report")?,
        max_rmse,
        fallback,
        worst_count,
    })
}

fn run_map(args: &Args) -> Result<(), String> {
    let palette = load_palette(&args.palette_path).map_err(|e| e.to_string())?;
    let config = MapConfig {
        max_rmse: args.max_rmse,
        fallback: args.fallback,
        gamma: 0.0,
    };
    let mapper = PaletteMapper::new(palette, &config)?;

    let input_paths = if args.input.is_dir() {
        collect_pngs(&args.input)?
    } else {
        vec![args.input.clone()]
    };
    if input_paths.is_empty() {
        return Err(format!("no PNG files found under {}", args.input.display()));
    }

    let images = input_paths
        .iter()
        .map(|p| load_image(p))
        .collect::<Result<Vec<_>, _>>()?;

    let root = if args.input.is_dir() {
        args.input.clone()
    } else {
        args.input
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };

    let report = run_batch(&mapper, &images, &root);

    for image in &images {
        let (indices, _) = mapper.map(&image.pixels, image.width, image.height);
        let composed = mapper.compose(&indices);
        let relative = image.path.strip_prefix(&root).unwrap_or(&image.path);
        let destination = args.output_dir.join(relative);
        save_rgba_png(&destination, &composed, image.width, image.height)?;
    }

    let worst = worst_samples(&report, args.worst_count);
    if let Some(parent) = args.report_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let file = std::fs::File::create(&args.report_path).map_err(|e| e.to_string())?;
    let mut writer = std::io::BufWriter::new(file);
    write_json_report(&mut writer, &report, &worst, mapper.palette().len())
        .map_err(|e| e.to_string())?;

    print_summary(&report, &worst, mapper.palette().len());
    Ok(())
}

fn print_summary(
    report: &palette_map::batch::BatchReport,
    worst: &[&palette_map::batch::ImageReport],
    palette_len: usize,
) {
    println!("Palette mapping complete");
    println!("  images:                 {}", report.image_count);
    println!("  palette entries:        {}", palette_len);
    println!("  total pixels:           {}", report.total_pixels);
    println!("  exact-match pixels:     {}", report.total_exact);
    println!("  fallback pixels:        {}", report.total_fallback);
    println!("  edge fringe hits:       {}", report.total_edge_fringe);
    println!("  mean RMSE:              {:.4}", report.mean_rmse_weighted);
    println!("  worst image mean RMSE:  {:.4}", report.worst_mean_rmse);
    println!("  worst pixel RMSE:       {:.4}", report.worst_max_rmse);
    println!("  worst samples:");
    for sample in worst.iter().take(5) {
        println!(
            "    {:<24} mean={:7.3} max={:7.3} fallback={}",
            sample.relative_path,
            sample.mean_rmse,
            sample.stats.max_dist_rmse,
            sample.stats.fallback_pixels
        );
    }
}

fn run_make_fixtures(output: &Path, count: usize) -> Result<(), String> {
    std::fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let palette = make_palette_256();
    write_palette_png(&output.join("palette256.png"), &palette)?;
    let texture_dir = output.join("textures");
    generate_fixtures(&texture_dir, count)?;

    // Also emit a text version to exercise the text loader in documentation.
    let mut text = String::new();
    for color in &palette {
        text.push_str(&format!(
            "#{:02x}{:02x}{:02x}{:02x}\n",
            color.r, color.g, color.b, color.a
        ));
    }
    std::fs::write(output.join("palette256.txt"), text).map_err(|e| e.to_string())?;
    println!(
        "Wrote {} palette entries and {} fixtures to {}",
        palette.len(),
        count,
        output.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv.first().map(String::as_str).unwrap_or("palette-map");

    let result = match argv.get(1).map(String::as_str) {
        Some("map") => match parse_args(&argv) {
            Ok(args) => run_map(&args),
            Err(err) => {
                eprintln!("error: {err}");
                print_usage(program);
                return ExitCode::FAILURE;
            }
        },
        Some("make-fixtures") => {
            let mut output = None;
            let mut count = 500usize;
            let mut iter = argv.iter().skip(2);
            while let Some(flag) = iter.next() {
                match flag.as_str() {
                    "--output" => output = iter.next().map(PathBuf::from),
                    "--count" => {
                        count = match iter.next().and_then(|v| v.parse().ok()) {
                            Some(v) => v,
                            None => {
                                eprintln!("error: --count requires a positive integer");
                                return ExitCode::FAILURE;
                            }
                        }
                    }
                    other => {
                        eprintln!("error: unknown argument: {other}");
                        print_usage(program);
                        return ExitCode::FAILURE;
                    }
                }
            }
            match output {
                Some(path) => run_make_fixtures(&path, count),
                None => {
                    eprintln!("error: make-fixtures requires --output <dir>");
                    print_usage(program);
                    return ExitCode::FAILURE;
                }
            }
        }
        Some("-h") | Some("--help") | None => {
            print_usage(program);
            Ok(())
        }
        Some(other) => {
            eprintln!("error: unknown subcommand: {other}");
            print_usage(program);
            return ExitCode::FAILURE;
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
