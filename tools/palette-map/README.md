# palette-map

Remap game textures onto a single fixed external 256-color palette.

## Guarantees

- **No new colors.** Every output pixel is a literal entry from the external
  palette. Output is composed by palette-index lookup, never by inventing or
  blending an RGB value.
- **Threshold fallback.** When the closest palette entry is farther than
  `--max-rmse`, the pixel is sent to either the palette's transparent entry
  (`--fallback transparent`, the default) or an explicitly named palette
  index (`--fallback-color <index>`).
- **Deterministic.** The nearest-neighbor decision uses a k-d tree whose
  ties break to the lowest palette index, plus a per-color decision cache.
  The same input always produces byte-identical output and reports, run
  after run and across platforms.
- **No edge fringe.** The distance is computed in the premultiplied-alpha
  linear space used by libimagequant, and fully transparent palette entries
  are excluded from the nearest-neighbor vote for any non-transparent input
  pixel. Anti-aliased sprite edges therefore keep a real (semi-transparent)
  palette color instead of being punched full of transparent holes.

## Distance and threshold

`FPixel::diff` is a direct copy of libimagequant's `f_pixel::diff`: channels
  are linearized with the sRGB gamma table, premultiplied by alpha and
  weighted, and each RGB channel takes the max of its on-black and on-white
  difference. The CLI threshold `--max-rmse` is expressed as the equivalent
  0..255-scale RMSE (mean of the three RGB squared channel errors), so a value
  of `30` means "reject matches whose average per-channel error exceeds ~30
  levels". Omit the flag to disable the threshold entirely.

## Palette formats

- **PNG** (`.png`): every distinct pixel color, in raster order, becomes an
  entry. A 16x16 image therefore yields up to 256 entries.
- **Text** (any other extension): one color per line as `#RGB`, `#RGBA`,
  `#RRGGBB` or `#RRGGBBAA`. A line whose only token is `#` is a comment.

Duplicate entries are removed while preserving first-seen order; the palette
must contain at most 256 colors and, when using transparent fallback, at
least one fully transparent entry.

## Usage

```bash
# Generate a 256-color palette plus 500 sample textures (for demos/CI).
cargo run -p palette-map --release -- \
  make-fixtures --output ./fixtures --count 500

# Map a whole directory of PNGs.
cargo run -p palette-map --release -- \
  map --palette ./fixtures/palette256.png \
      --input ./fixtures/textures \
      --output ./build/mapped \
      --report ./build/report.json \
      --max-rmse 30 --worst 10

# Map a single file, rejecting bad matches to palette index 7.
cargo run -p palette-map --release -- \
  map --palette palette.txt --input skin.png --output out/ \
      --report report.json --fallback-color 7
```

## Report

`report.json` contains:

- `summary`: image/pixel totals, exact-match and fallback counts, edge fringe
  hits, weighted mean RMSE, worst image mean RMSE and worst single-pixel RMSE;
- `worst_samples`: the `--worst` highest-error images with p95/p99/max RMSE;
- `images`: a full per-image record in stable sorted order.

## Tests

```bash
cargo test -p palette-map
```

The suite cross-validates nearest-neighbor output against libimagequant's own
`QuantizationResult::from_palette`, asserts pixel-exact determinism over
repeated runs, checks that no output color exists outside the palette, and
covers both the clean-edge path and the edge-fringe detector.
