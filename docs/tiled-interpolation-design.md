# Tiled RIFE interpolation — design + prototype

**Goal.** Make RIFE 4.25 frame interpolation admissible at 4K (and beyond) without
exceeding the Metal working set, by bounding peak activation to a per-tile budget instead
of letting it scale linearly with input resolution.

**Status.** Design + working prototype (`RIFEPairTiler` in
`Sources/MLXRIFE/RIFETileProcessor.swift`, wired into `RIFEInterpolatePackage`). Seam
evaluation methodology specified; numeric seam sweep is the remaining follow-up (needs the
4K A/B corpus + Metal).

---

## 1. The problem, measured

RIFE has **no tiling** today: the whole frame is resident through the coarse-to-fine
pyramid, so peak activation is resolution-**linear**. Re-baselined 2026-07-01 against real
`phys_footprint` on the video path (factor 2):

| Input | Pixels | Peak activation |
|------:|-------:|----------------:|
| 360p  | 0.23 Mpx | 0.77 GB |
| 720p  | 0.92 Mpx | 2.37 GB |
| 1080p | 2.07 Mpx | 3.86 GB |
| **2160p (4K)** | **8.29 Mpx** | **~15 GB (extrapolated)** |

The slope is ~1.86 GB/Mpx. 4K extrapolates to ~15 GB, which exceeds
`maxRecommendedWorkingSetSize` on most Macs (and blows the manifest's declared 4.0 GB
`peakActivationBytes`, which is only valid to 1080p). Interpolation at 4K is currently
inadmissible.

## 2. Why RIFE tiling is harder than SR tiling

SeedVR2 / Real-ESRGAN already tile with `MLXTileProcessor(tileSize:overlap:scale:)`:
split the frame into overlapping tiles, run each independently, feather-blend the seams.
That works because SR is a **spatially local, single-input** op — a tile's output depends
only on nearby pixels, so a small feather overlap (SeedVR2 uses 32 px) hides the seam.

RIFE breaks two of those assumptions:

1. **Two inputs, one output.** Interpolation is `(frame0, frame1) → midpoint`.
   `MLXTileProcessor.process(_:forward:)` takes a *single* input buffer. A RIFE tiler must
   extract **co-located** tiles from *both* frames with identical geometry and run the pair
   through the model together.

2. **Optical flow is non-local.** RIFE's IFNet
   (`Sources/RIFEMLX/IFNet.swift`) estimates a bidirectional flow field over a 5-level
   pyramid (`scaleList = [16, 8, 4, 2, 1]`), warps both frames toward the midpoint, and
   blends with a learned mask. To interpolate a pixel *p* correctly, the tile must contain
   *p*'s correspondence in **both** frames — i.e. both *p* and *p ± motion*. If an object
   moves across a tile boundary by more than the overlap, the per-tile flow can't see where
   it came from and produces a wrong warp → a visible seam / ghost exactly where motion
   crosses the cut. This is the classic tiled-optical-flow failure and is the real design
   constraint.

So the RIFE overlap is not a cosmetic feather band — it is a **motion context halo**:
extra input that the model *computes over* so that flow near the kept interior is correct,
most of which is then discarded / down-weighted at blend time.

## 3. Memory model — why tiling bounds it

IFNet's activation is dominated by the per-block conv stacks (`c = [192,128,96,64,32]`,
each block = 2 stride-2 convs + 8 residual convs) plus the full-resolution warps and the
final merge. Every one of those tensors is proportional to the tile's pixel count. Because
each intermediate is `MLX.eval`'d and pulled to the CPU **per tile**, the MLX buffer pool
stabilises at roughly **one tile's** activation and is reused across tiles (same tile size
→ same allocation pattern). Peak activation therefore becomes a function of *tile* size, not
*frame* size — exactly how SeedVR2 keeps a constant 4.5 GB across arbitrary input
resolution.

**Tile budget.** Pick the tile so per-tile pixels ≤ 1080p pixels; then the already-measured
**3.86 GB @1080p is the ceiling** and the declared 4.0 GB stays honest at any input
resolution. Default `tileSize = 1024` (1.05 Mpx, ~1.95 GB per tile) leaves comfortable
headroom for the resident weights (0.04 GB), the two full-frame NHWC inputs (~0.2 GB at 4K),
and the blend accumulator (~0.1 GB). **Measured 2026-07-01: 3.02 GB at true 3840×2160** (the
~2.3 GB estimate was low; see §9 Finding 4) — still comfortably under the declared 4.0 GB.

Tile counts (`step = tileSize − overlap`, edge origins clamped inside the frame):

| tileSize / overlap | 4K tiles | per-tile peak | trade |
|---|---:|---:|---|
| 1024 / 128 | 5×3 = 15 | ~1.95 GB | safest memory, most dispatches |
| 1280 / 128 | 4×2 = 8  | ~3.05 GB | balanced |
| 1440 / 128 | 3×2 = 6  | ~3.86 GB | at budget, fewest dispatches |

We default to **1024 / 128** and expose all three knobs in `RIFEConfiguration`.

## 4. Overlap (halo) sizing — the crux

The halo must cover the largest inter-frame motion we want to resolve correctly near a kept
interior boundary: **overlap ≥ P99 per-frame displacement**.

Two things make this tractable:

- **Interpolation is between *adjacent* frames**, so temporal delta is small. Typical action
  motion is 10–60 px/frame at 4K; a full-screen 1 s pan at 24 fps is ~160 px/frame (the
  pathological tail).
- **RIFE's coarse pyramid gives per-tile robustness.** The coarsest block sees the tile
  downsampled 16× (a 1024 tile → 64 px), and its conv receptive field spans most of that, so
  a ~1024 tile retains roughly the same motion-handling capacity RIFE has on a whole 1080p
  frame. Tiling at ~1024 does **not** meaningfully shrink the model's intrinsic motion range;
  it only requires the halo to carry the correspondence for boundary pixels.

**Recommendation:** default `overlap = 128` px (covers ~128 px/frame motion, generous for
typical 4K content). Because the feather ramp width equals `overlap`, adjacent tiles
cross-blend their degraded edges against each other's good interiors, which further hides
residual flow discontinuity. For pathological large motion, the fallback is the existing
**`scale` knob** (§5), not a bigger halo (a halo large enough for 160 px motion erodes the
memory win).

## 5. Interaction with the existing `scale` knob (and why it isn't enough alone)

`RIFEConfiguration.scale` (< 1 for high-res) divides the pyramid resolution so the finest
blocks run downsampled — Practical-RIFE recommends 0.5 for 2K/4K. But in `IFNet` the
Head encoder, the per-level warps, and the **final full-resolution merge** always run at
native H×W regardless of `scale`; only the internal IFBlock convs shrink. So global
`scale = 0.5` reduces but does **not** quarter peak activation, and 4K would still land well
above 4 GB. Global downscale also estimates flow below native resolution, losing fine-motion
detail everywhere.

Tiling is complementary and strictly better for quality: each tile is ≤ ~1080p, so we can
keep **`scale = 1.0` inside tiles** (native-resolution flow) and still bound memory — the one
thing whole-frame downscaling can't do. `scale < 1` remains available *within* tiles as the
escape hatch for pathological cross-seam motion.

## 6. Reuse decision — vendor, don't import

`MLXTileProcessor` lives in `RealESRGANMLX` and is single-input; it also references
`PlaybackTierError` (SR-internal). Pulling the whole `mlx-realesrgan-swift` package into RIFE
just to get a tiler — then still needing a two-input variant — is the wrong dependency.

Per CLAUDE.md ("vendor only what's needed"), the prototype adds a small **`RIFEPairTiler`**
in `Sources/MLXRIFE`, borrowing `MLXTileProcessor`'s proven geometry (clamped tile origins,
`step = tileSize − overlap`, feather ramp = `overlap`) but:

- taking **two** co-located NHWC inputs and a `(tileA, tileB) → tile` closure;
- blending via **weighted accumulation** (`accum += w·val`, `wsum += w`, final `accum/wsum`)
  instead of in-place alpha, so the result is order-independent;
- operating in NHWC `MLXArray` space so it reuses the package's existing
  `rgbNHWC` / `pixelBuffer(fromRGBNHWC:)` converters unchanged — the tiler slices the two
  full-frame arrays, runs the model per tile-pair, and returns a single `[1,H,W,3]` array
  that flows into the existing pixel-buffer path.

If a third caller ever needs pair-tiling, promote `RIFEPairTiler` to a shared module
alongside `MLXTileProcessor` (Layer-2 consolidation, already flagged in the converter
comments).

## 7. Prototype

`Sources/MLXRIFE/RIFETileProcessor.swift` — `RIFEPairTiler(tileSize:overlap:)` with
`interpolate(img0:img1:run:)`. Wired into `RIFEInterpolatePackage.run`: the per-intermediate
step branches on input pixel count —

```
if tileThresholdPixels > 0 && H*W > tileThresholdPixels:
    RIFEPairTiler(tileSize, overlap).interpolate(img0, img1) { ta, tb in
        model.inference(img0: ta, img1: tb, timestep: t, scale: scale) }   // eval per tile
else:
    model.inference(img0, img1, timestep: t, scale: scale)                 // whole-frame
```

The ≤1080p whole-frame path is byte-for-byte unchanged (threshold default = 1920·1080), so
existing parity is preserved; tiling only engages above 1080p. Knobs land in
`RIFEConfiguration`: `tileThresholdPixels` (2_073_600), `tileSize` (1024), `tileOverlap`
(128).

## 8. Manifest change

Tiling makes `peakActivationBytes` **resolution-independent** (like SeedVR2). Because the
default tile is ≤1080p, the measured **3.86 GB @1080p is the per-tile ceiling**, so the
declared **4.0 GB stays valid — now at any input resolution including 4K**. The manifest
comment is updated from "declared for a 1080p max input" to note the tiled bound and that
4K+ is admissible. No numeric change to `peakActivationBytes` is required; the deliverable's
"raise supported resolution" is realised by removing the 1080p cap on validity, not by
raising the byte figure.

## 9. Seam-artifact evaluation — RESULTS (BRIDGE-VID-003, run 2026-07-01)

Run on M-series Metal via the `RIFESeamEval` executable target (whole-frame ref vs tiled, same
pairs; PNG A/B into ffmpeg for SSIM/VMAF; along-seam error = tiled-vs-ref over pixels the grid
covers ≥2×). Corpus: three real 4K clips — a slow/global-motion graphics clip (true 3840×2160), a
near-static clip, and a people/sports clip with a real fast-motion tail (3240×1920). Whole-frame
4K reference fit at `scale=1` on a 128 GB box, so the reference is EXACT.

**Finding 1 — three motion regimes (tiled vs whole-frame, `tileSize=1024`, `scale=1`):**

| Regime | example | overlap 32 → 256 (global PSNR) | verdict |
|---|---|---|---|
| Normal (adjacent frames) | any clip, 1080p & 4K | 58–80 dB, flat in overlap | near-lossless; overlap irrelevant |
| Resolvable large motion | fast-tail adjacent pair | 42.4 → 45.6 → 46.9 dB | **clean knee at overlap ≈ 128** (256 = +1.3 dB only) |
| Beyond RIFE's range | ~20-frame synthetic gap | 27–37 dB, non-monotonic | model-range limit, not a tiling artifact |

**Finding 2 — perceptual (tiled ov128 vs whole-frame ref, ffmpeg):** SSIM/VMAF are transparent
even under fast motion — resolvable-motion clip **0.995 / 95.9**, slow-motion clip **0.996 / 96.9**,
near-static clip **0.994 / 97.4**. `overlap=256` adds only +0.1 VMAF. ⇒ **overlap=128 is perceptually
indistinguishable from whole-frame.**

**Finding 3 — `scale=0.5` does NOT help and usually HURTS.** On resolvable large motion it was
~10 dB WORSE than `scale=1` (fast-tail global 34–36 dB vs 42–47 dB); on the beyond-range gap it
recovered only marginally (≈+2 dB). Downscaling the flow loses the fine motion that native-res
tiling resolves correctly — consistent with §5's thesis (tiling lets you KEEP `scale=1`). ⇒ the
"`scale=0.5` fast-pan escape hatch" anticipated here is retracted: **overlap is the lever; keep
`scale=1`.** The genuine escape hatch for pathological (beyond-range) motion is a global flow prior
(§10), not downscaling.

**Finding 4 — 4K memory (Task C).** In-app `phys_footprint`, factor-2, `tileSize=1024/overlap=128`:
whole-frame 4K ref **13.67 GB** (confirms the ~15 GB extrapolation); **tiled = 3.02 GB** at true
3840×2160 (2.83 GB at 3240×1920). **≤ the declared 4.0 GB `peakActivationBytes`** ✓, with margin.
(NB: the first tiled run after a whole-frame ref reads high — ~11 GB — until MLX reclaims the ref's
pool; the settled per-tile working set is ~3 GB. Measure Task C without a preceding whole-frame ref
for the cleanest number.) The design's ~2.3 GB estimate was low; actual ~3.0 GB — still under budget.

**Locked defaults (validated, unchanged):** `tileSize=1024`, `overlap=128`, `scale=1.0`. 1440 was
not needed — 1024 already fits 4K with margin and is the safest; 1440 (≈3.86 GB/tile) narrows the
headroom for no quality need. No `RIFEConfiguration` change required.

Harness: `PROD/mlx-rife-swift` `RIFESeamEval` executable target (added 2026-07-01) —
`swift run -c release RIFESeamEval --a f0.png --b f1.png --out DIR --tile-size N --overlaps a,b,c --scale s`.

## 10. Follow-ups / not done here

- Run the §9 sweep and lock the default overlap on data.
- Optional: a low-res **global flow prior** seeded into `block0` to make per-tile flow
  agree across seams under large motion (removes the fast-pan fallback, but is a real IFNet
  change — deferred).
- Speed: the CPU `rgbNHWC` conversion is a per-pixel loop (~8.3 Mpx ×2 at 4K) and tiling adds
  dispatches; memory, not throughput, is the target here, but both are worth profiling
  (`MLX_PROFILE=1`, the region hooks are already in place).
