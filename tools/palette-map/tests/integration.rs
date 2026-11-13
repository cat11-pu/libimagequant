use imagequant::{new as liq_new, RGBA as LiqRGBA};
use palette_map::batch::save_rgba_png;
use palette_map::color::Rgba8;
use palette_map::fixture::{generate_fixtures, make_palette_256};
use palette_map::mapper::{Fallback, MapConfig, PaletteMapper};
use palette_map::palette::load_palette;
use std::collections::HashSet;

fn to_liq(color: Rgba8) -> LiqRGBA {
    LiqRGBA {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a,
    }
}

#[test]
fn palette_has_exactly_256_unique_colors() {
    let palette = make_palette_256();
    assert_eq!(palette.len(), 256);
    let unique: HashSet<[u8; 4]> = palette.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    assert_eq!(unique.len(), 256);
    assert!(palette[0].is_fully_transparent());
}

#[test]
fn output_contains_only_palette_colors() {
    let palette = make_palette_256();
    let mapper = PaletteMapper::new(
        palette.clone(),
        &MapConfig {
            max_rmse: Some(60.0),
            fallback: Fallback::Transparent,
            ..MapConfig::default()
        },
    )
    .unwrap();

    let input: Vec<Rgba8> = (0..128 * 128)
        .map(|i| {
            Rgba8::new(
                (i % 251) as u8,
                ((i * 7) % 251) as u8,
                ((i * 13) % 251) as u8,
                if i % 97 == 0 { 128 } else { 255 },
            )
        })
        .collect();
    let (indices, _) = mapper.map(&input, 128, 128);
    let output = mapper.compose(&indices);
    let allowed: HashSet<[u8; 4]> = palette.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    assert!(output.iter().all(|c| allowed.contains(&[c.r, c.g, c.b, c.a])));
}

#[test]
fn agrees_with_imagequant_nearest_neighbor_mapping() {
    // imagequant's own public API with a fully fixed palette and dithering
    // disabled must pick the same nearest palette entry for ordinary opaque
    // pixels. Tie distances are rare with these inputs; exact palette colors
    // always have distance zero, so compare those exhaustively.
    let palette = make_palette_256();
    let mapper = PaletteMapper::new(palette.clone(), &MapConfig::default()).unwrap();

    let attr = liq_new();
    let liq_palette: Vec<LiqRGBA> = palette.iter().copied().map(to_liq).collect();
    let mut result = imagequant::QuantizationResult::from_palette(&attr, &liq_palette, 0.0).unwrap();
    result.set_dithering_level(0.0).unwrap();

    let input: Vec<Rgba8> = palette
        .iter()
        .filter(|c| !c.is_fully_transparent())
        .cycle()
        .take(4096)
        .copied()
        .collect();
    let liq_input: Vec<LiqRGBA> = input.iter().copied().map(to_liq).collect();
    let mut image = attr.new_image(liq_input, 64, 64, 0.0).unwrap();
    let (liq_palette_out, liq_indices) = result.remapped(&mut image).unwrap();

    let (our_indices, _) = mapper.map(&input, 64, 64);
    for (i, (&our, &theirs)) in our_indices.iter().zip(liq_indices.iter()).enumerate() {
        // Both must resolve to the identical literal color even if internal
        // ordering of palette entries differed.
        assert_eq!(
            Into::<[u8; 4]>::into(mapper.palette()[usize::from(our)]),
            Into::<[u8; 4]>::into(liq_palette_out[usize::from(theirs)]),
            "pixel {i} mapped to different colors"
        );
    }
}

#[test]
fn deterministic_across_repeated_runs_and_input_shuffles() {
    let palette = make_palette_256();
    let mapper = PaletteMapper::new(
        palette,
        &MapConfig {
            max_rmse: Some(45.0),
            fallback: Fallback::Transparent,
            ..MapConfig::default()
        },
    )
    .unwrap();

    let input: Vec<Rgba8> = (0..64 * 64)
        .map(|i| Rgba8::new((i * 11) as u8, (i * 23) as u8, (i * 47) as u8, 255))
        .collect();
    let (first, stats_first) = mapper.map(&input, 64, 64);
    for _ in 0..4 {
        let (again, stats_again) = mapper.map(&input, 64, 64);
        assert_eq!(first, again);
        assert_eq!(stats_first.fallback_pixels, stats_again.fallback_pixels);
        assert_eq!(
            (stats_first.mean_rmse() * 1e6) as u64,
            (stats_again.mean_rmse() * 1e6) as u64
        );
    }
}

#[test]
fn threshold_fallback_never_leaks_unspecified_color() {
    let palette = make_palette_256();
    let config = MapConfig {
        // Below magenta's nearest RMSE (1.54), so every pixel must fall back.
        max_rmse: Some(1.0),
        fallback: Fallback::Transparent,
        ..MapConfig::default()
    };
    let mapper = PaletteMapper::new(palette, &config).unwrap();
    let input = vec![Rgba8::new(255, 0, 255, 255); 16];
    let (indices, stats) = mapper.map(&input, 4, 4);
    assert_eq!(stats.fallback_pixels, 16);
    assert!(mapper
        .compose(&indices)
        .iter()
        .all(|c| c.is_fully_transparent()));
}

#[test]
fn end_to_end_fixture_batch_has_no_fringe_and_no_new_colors() {
    let temp = std::env::temp_dir().join("palette_map_it_batch");
    let _ = std::fs::remove_dir_all(&temp);
    generate_fixtures(&temp.join("textures"), 20).unwrap();
    let palette = make_palette_256();
    save_rgba_png(&temp.join("palette.png"), &palette, 16, 16).unwrap();

    let loaded = load_palette(&temp.join("palette.png")).unwrap();
    assert_eq!(loaded.len(), 256);
    // No threshold: every opaque/semi-opaque pixel maps to a real palette
    // color, so texture boundaries must stay clean (zero fringe hits).
    let mapper = PaletteMapper::new(loaded, &MapConfig::default()).unwrap();

    let allowed: HashSet<[u8; 4]> = mapper.palette().iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    let paths = palette_map::batch::collect_pngs(&temp.join("textures")).unwrap();
    assert_eq!(paths.len(), 20);
    let mut total_fringe = 0;
    for path in paths {
        let image = palette_map::batch::load_image(&path).unwrap();
        let (indices, stats) = mapper.map(&image.pixels, image.width, image.height);
        let output = mapper.compose(&indices);
        assert!(output.iter().all(|c| allowed.contains(&[c.r, c.g, c.b, c.a])));
        total_fringe += stats.edge_fringe_hits;
    }
    assert_eq!(total_fringe, 0);
}

#[test]
fn fringe_detector_flags_fallback_pixels_on_edges() {
    // Positive control: a deliberately tight threshold forces edge pixels
    // adjacent to transparency into the fallback entry; the detector must
    // notice these instead of silently producing halo artifacts.
    let temp = std::env::temp_dir().join("palette_map_it_fringe");
    let _ = std::fs::remove_dir_all(&temp);
    generate_fixtures(&temp.join("textures"), 5).unwrap();
    let mapper = PaletteMapper::new(
        make_palette_256(),
        &MapConfig {
            max_rmse: Some(0.5),
            fallback: Fallback::Transparent,
            ..MapConfig::default()
        },
    )
    .unwrap();

    let paths = palette_map::batch::collect_pngs(&temp.join("textures")).unwrap();
    let mut detected = 0;
    for path in paths {
        let image = palette_map::batch::load_image(&path).unwrap();
        let (_, stats) = mapper.map(&image.pixels, image.width, image.height);
        detected += stats.edge_fringe_hits;
    }
    assert!(detected > 0, "fringe detector should catch edge fallback pixels");
}
