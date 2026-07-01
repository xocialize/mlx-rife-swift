import Testing
import Foundation
@testable import MLXRIFE

/// Offline geometry / blend correctness for `RIFEPairTiler` — no model, no weights, and no MLX
/// evaluation (the CLI test host has no metallib; live interpolation is proven in the app). We
/// test the two pure-Swift crux pieces directly: the tile grid and the weighted-accumulation
/// blend.
struct RIFEPairTilerTests {

    // MARK: tile grid

    @Test func gridCoversEveryPixelAndClampsInward() {
        let H = 37, W = 43   // not a multiple of tileSize → forces clamped edge tiles
        let tiler = RIFEPairTiler(tileSize: 16, overlap: 6)
        let tiles = tiler.tileGrid(H: H, W: W)

        // Every tile stays inside the frame.
        for t in tiles {
            #expect(t.x >= 0 && t.y >= 0)
            #expect(t.x + t.tw <= W)
            #expect(t.y + t.th <= H)
        }
        // Every pixel is covered by ≥1 tile (else the blend would divide by zero there).
        var covered = [Bool](repeating: false, count: H * W)
        for t in tiles {
            for yy in t.y ..< (t.y + t.th) {
                for xx in t.x ..< (t.x + t.tw) { covered[yy * W + xx] = true }
            }
        }
        #expect(covered.allSatisfy { $0 })
    }

    @Test func gridIsSinglePassWhenFrameSmallerThanTile() {
        let tiler = RIFEPairTiler(tileSize: 64, overlap: 16)
        let tiles = tiler.tileGrid(H: 8, W: 8)
        #expect(tiles == [RIFEPairTiler.Tile(x: 0, y: 0, tw: 8, th: 8)])
    }

    // MARK: weighted-accumulation blend

    /// If every tile carries the same underlying frame values, the normalised blend
    /// (Σ w·val / Σ w) must reconstruct that frame EXACTLY regardless of feather weights or how
    /// many overlapping tiles touch a pixel. This is the property that guarantees seams don't
    /// shift color even before any RIFE numerics are involved.
    @Test func overlappingPassThroughTilesReconstructSourceExactly() {
        let H = 37, W = 43
        let tiler = RIFEPairTiler(tileSize: 16, overlap: 6)

        // Deterministic distinct value per pixel/channel so a mis-stitch can't alias.
        var src = [Float](repeating: 0, count: H * W * 3)
        for i in 0 ..< src.count { src[i] = Float(i % 251) / 251.0 }

        var accum = [Float](repeating: 0, count: H * W * 3)
        var wsum = [Float](repeating: 0, count: H * W)

        for t in tiler.tileGrid(H: H, W: W) {
            // Extract this tile's slice of src (what a pass-through forward would return).
            var vals = [Float](repeating: 0, count: t.tw * t.th * 3)
            for ty in 0 ..< t.th {
                for tx in 0 ..< t.tw {
                    let g = ((t.y + ty) * W + (t.x + tx)) * 3
                    let s = (ty * t.tw + tx) * 3
                    vals[s + 0] = src[g + 0]; vals[s + 1] = src[g + 1]; vals[s + 2] = src[g + 2]
                }
            }
            tiler.blend(vals: vals, x: t.x, y: t.y, tw: t.tw, th: t.th,
                        W: W, ov: 6, accum: &accum, wsum: &wsum)
        }

        var maxErr: Float = 0
        for p in 0 ..< (H * W) {
            let w = wsum[p]
            #expect(w > 0)
            for c in 0 ..< 3 { maxErr = max(maxErr, abs(accum[p * 3 + c] / w - src[p * 3 + c])) }
        }
        #expect(maxErr < 1e-5)
    }
}
