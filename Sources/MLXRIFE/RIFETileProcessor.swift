//
//  RIFETileProcessor.swift
//  MLXRIFE
//
//  Memory-bounded pair-tiler for RIFE frame interpolation. RIFE has no internal tiling, so
//  peak activation is resolution-LINEAR (~1.86 GB/Mpx) — 4K extrapolates to ~15 GB and blows
//  the Metal working set. This splits BOTH input frames into co-located overlapping tiles,
//  runs each tile-PAIR through the model independently, and feather-blends the interpolated
//  output tiles. Because each tile is evaluated and pulled to the CPU before the next, the MLX
//  buffer pool stabilises at ~one tile's activation and is reused across tiles — so peak
//  activation becomes a function of TILE size, not frame size.
//
//  Why a bespoke tiler and not RealESRGANMLX.MLXTileProcessor:
//    - MLXTileProcessor is SINGLE-input (SR: one frame in, one up-scaled frame out). RIFE is a
//      TWO-input op: (frame0, frame1) → midpoint, and both tiles must share identical geometry.
//    - Pulling in the whole mlx-realesrgan-swift package (and its SR-internal error type) just
//      to get a tiler is the wrong dependency. Per CLAUDE.md ("vendor only what's needed") we
//      reuse MLXTileProcessor's proven geometry (clamped origins, step = tileSize − overlap,
//      feather ramp = overlap) in a small pair-aware helper that stays in NHWC MLXArray space,
//      so the package's existing rgbNHWC / pixelBuffer converters are reused unchanged.
//
//  The overlap is a MOTION CONTEXT HALO, not a cosmetic feather: RIFE estimates optical flow,
//  and a tile can only interpolate a pixel correctly if the tile contains that pixel's
//  correspondence in both frames. overlap must therefore cover the expected per-frame motion
//  near the kept interior boundary. See docs/tiled-interpolation-design.md §4.
//

import Foundation
import MLX

/// Splits two co-located NHWC frames into overlapping tiles, runs each tile-pair through a
/// supplied RIFE forward, and feather-blends the interpolated output tiles into one frame.
///
/// Blending uses weighted accumulation (`accum += w·val`, `wsum += w`, final `accum / wsum`)
/// so the result is independent of tile visit order. The feather weight is 1.0 across the tile
/// interior and ramps to ~0 over `overlap` pixels at each edge — identical in spirit to
/// `MLXTileProcessor.writeTile`, so seams match the SR path's look.
public struct RIFEPairTiler: @unchecked Sendable {

    public let tileSize: Int
    public let overlap: Int

    /// - Parameters:
    ///   - tileSize: side length of each square tile (including the context halo). Choose so a
    ///     tile's pixel count stays within the activation budget (default 1024 ≈ 1.05 Mpx,
    ///     ~1.95 GB, comfortably under the 3.86 GB measured at 1080p).
    ///   - overlap: halo / feather width in pixels. Must cover expected per-frame motion near
    ///     tile boundaries (default 128).
    public init(tileSize: Int, overlap: Int) {
        self.tileSize = max(tileSize, 1)
        self.overlap = max(overlap, 0)
    }

    /// Interpolate a full `[1, H, W, 3]` midpoint frame by tiling.
    ///
    /// - Parameters:
    ///   - img0, img1: `[1, H, W, 3]` NHWC RGB in [0, 1] (the two source frames).
    ///   - run: forward for one tile-pair. Receives co-located `[1, th, tw, 3]` tiles and must
    ///     return the interpolated `[1, th, tw, 3]` tile. The closure is responsible for
    ///     `MLX.eval`-ing its result before returning (MLX is lazy — an un-evaluated tensor
    ///     materialises as zeros; the silent killer from mlx-porting). Evaluating here is also
    ///     what bounds memory: it forces each tile's graph to run and free before the next.
    /// - Returns: `[1, H, W, 3]` NHWC RGB, the blended interpolated frame.
    /// One tile's placement in the frame: origin `(x, y)` and clamped size `(tw, th)`.
    struct Tile: Equatable { let x, y, tw, th: Int }

    /// The tile grid over an `H × W` frame: fixed-step origins with each origin clamped inward
    /// so the last row/column stays inside the frame (matches `MLXTileProcessor`). Every pixel
    /// is covered by ≥1 tile. Pure geometry — no MLX, unit-testable offline.
    func tileGrid(H: Int, W: Int) -> [Tile] {
        let step = max(tileSize - overlap, 1)
        var tiles: [Tile] = []
        for tileY in stride(from: 0, to: H, by: step) {
            for tileX in stride(from: 0, to: W, by: step) {
                let x = min(tileX, max(0, W - tileSize))
                let y = min(tileY, max(0, H - tileSize))
                tiles.append(Tile(x: x, y: y,
                                  tw: min(tileSize, W - x), th: min(tileSize, H - y)))
            }
        }
        return tiles
    }

    public func interpolate(
        img0: MLXArray,
        img1: MLXArray,
        run: (MLXArray, MLXArray) throws -> MLXArray
    ) rethrows -> MLXArray {
        let H = img0.shape[1]
        let W = img0.shape[2]
        let ov = Float(max(overlap, 1))

        // Blend accumulators (CPU). accum holds Σ w·val per channel; wsum holds Σ w per pixel.
        var accum = [Float](repeating: 0, count: H * W * 3)
        var wsum = [Float](repeating: 0, count: H * W)

        for tile in tileGrid(H: H, W: W) {
            let (x, y, tw, th) = (tile.x, tile.y, tile.tw, tile.th)
            let a = img0[0..., y ..< (y + th), x ..< (x + tw), 0...]
            let b = img1[0..., y ..< (y + th), x ..< (x + tw), 0...]

            let outTile = try run(a, b)
            // Defensive re-eval: `run` is contractually expected to eval, but a stray lazy graph
            // must never silently zero the read-out.
            MLX.eval(outTile)
            let vals = outTile.asType(.float32).asArray(Float.self)
            guard vals.count >= tw * th * 3 else { continue }

            blend(vals: vals, x: x, y: y, tw: tw, th: th,
                  W: W, ov: ov, accum: &accum, wsum: &wsum)
        }

        // Normalise Σ w·val by Σ w. Every pixel is covered by ≥1 tile (the grid spans the frame
        // and edge origins are clamped inward), and the feather weight is floored > 0, so wsum
        // is always positive — but guard anyway.
        var out = [Float](repeating: 0, count: H * W * 3)
        out.withUnsafeMutableBufferPointer { o in
            accum.withUnsafeBufferPointer { acc in
                wsum.withUnsafeBufferPointer { ws in
                    for p in 0 ..< (H * W) {
                        let w = ws[p]
                        let inv = w > 0 ? 1.0 / w : 0
                        o[p * 3 + 0] = acc[p * 3 + 0] * inv
                        o[p * 3 + 1] = acc[p * 3 + 1] * inv
                        o[p * 3 + 2] = acc[p * 3 + 2] * inv
                    }
                }
            }
        }
        return MLXArray(out, [1, H, W, 3])
    }

    /// Accumulate one interpolated tile into the frame with a feathered weight.
    ///
    /// Weight is `min(rampLeft, rampRight, rampTop, rampBottom)` where each ramp rises 0→1 over
    /// `overlap` px from its edge — 1.0 across the interior, tapering at the halo so a tile's
    /// (flow-degraded) edge is down-weighted against the neighbour tile's good interior.
    func blend(
        vals: [Float],
        x: Int, y: Int, tw: Int, th: Int,
        W: Int, ov: Float,
        accum: inout [Float], wsum: inout [Float]
    ) {
        vals.withUnsafeBufferPointer { v in
            accum.withUnsafeMutableBufferPointer { acc in
                wsum.withUnsafeMutableBufferPointer { ws in
                    for ty in 0 ..< th {
                        let wyTop = min(Float(ty) / ov, 1.0)
                        let wyBot = min(Float(th - 1 - ty) / ov, 1.0)
                        let wy = min(wyTop, wyBot)
                        let gy = y + ty
                        for tx in 0 ..< tw {
                            let wxLeft = min(Float(tx) / ov, 1.0)
                            let wxRight = min(Float(tw - 1 - tx) / ov, 1.0)
                            let w = max(min(min(wxLeft, wxRight), wy), 1e-3)

                            let gp = gy * W + (x + tx)
                            let s = (ty * tw + tx) * 3
                            acc[gp * 3 + 0] += v[s + 0] * w
                            acc[gp * 3 + 1] += v[s + 1] * w
                            acc[gp * 3 + 2] += v[s + 2] * w
                            ws[gp] += w
                        }
                    }
                }
            }
        }
    }
}
