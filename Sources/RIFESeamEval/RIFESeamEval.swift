// RIFESeamEval — BRIDGE-VID-003 seam-artifact + 4K memory harness for tiled RIFE interpolation.
//
// Runs one adjacent frame-pair (frame0, frame1) two ways and compares:
//   • REFERENCE — whole-frame `RIFEModel.inference` (no tiling; the exact baseline).
//   • TILED     — `RIFEPairTiler(tileSize, overlap).interpolate` for each overlap in the sweep.
// then reports, per overlap:
//   • along-SEAM error  — tiled-vs-ref MAE/PSNR over pixels the tile grid covers ≥2× (the halo
//     bands = exactly where the tiling seam artifact lives). Isolates the seam from global quality.
//   • GLOBAL error      — tiled-vs-ref MAE/PSNR over the whole frame.
//   • peak phys_footprint over the run (Task C: the real admission basis, not an MLX smoke peak).
// PNGs (ref + each tiled) are written so ffmpeg can compute PSNR/SSIM/VMAF out-of-process.
//
// Why an executable, not a test: this needs live Metal (the `swift test` host lacks the metallib).
// Run: `swift run -c release RIFESeamEval --a f0.png --b f1.png --out /tmp/rife --overlaps 16,32,64,128`
// (or via xcodebuild, per the LTX GPU-watchdog lesson). Frame extraction is done upstream by ffmpeg.

import Foundation
import Darwin
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import RIFEMLX
import MLXRIFE

// MARK: - args

func argValue(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

let aPath = argValue("--a") ?? { fatalError("--a <frame0.png> required") }()
let bPath = argValue("--b") ?? { fatalError("--b <frame1.png> required") }()
let outDir = argValue("--out") ?? "/tmp/rife_seam_eval"
let tag = argValue("--tag") ?? "clip"
let tileSize = argValue("--tile-size").flatMap { Int($0) } ?? 1024
let overlaps = (argValue("--overlaps") ?? "16,32,64,128").split(separator: ",").compactMap { Int($0) }
let scale = argValue("--scale").flatMap { Float($0) } ?? 1.0
let weightsPath = argValue("--weights")
    ?? "\(NSHomeDirectory())/Documents/huggingface/models/mlx-community/RIFE-4.25/model.safetensors"

try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

// MARK: - phys_footprint (OS working set — the R-MEM-1 admission basis)

func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

/// Background peak sampler (100 ms) — captures the working-set high-water across a GPU run.
final class PhysSampler: @unchecked Sendable {
    private let q = DispatchQueue(label: "phys.sampler")
    private var running = false
    private(set) var peak: UInt64 = 0
    func start() {
        peak = physFootprintBytes(); running = true
        q.async { [self] in while running { peak = max(peak, physFootprintBytes()); usleep(100_000) } }
    }
    func stop() -> UInt64 { running = false; peak = max(peak, physFootprintBytes()); return peak }
}

// MARK: - PNG I/O  (raw 8-bit / 255 → [1,H,W,3], matching the package's rgbNHWC convention)

func loadRGB(_ path: String) -> (px: [Float], h: Int, w: Int)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = img.width, h = img.height
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    var px = [Float](repeating: 0, count: w * h * 3)
    for i in 0 ..< (w * h) {
        px[i * 3 + 0] = Float(rgba[i * 4 + 0]) / 255.0
        px[i * 3 + 1] = Float(rgba[i * 4 + 1]) / 255.0
        px[i * 3 + 2] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return (px, h, w)
}

func savePNG(_ px: [Float], h: Int, w: Int, to path: String) {
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< (w * h) {
        @inline(__always) func c(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }
        rgba[i * 4 + 0] = c(px[i * 3 + 0]); rgba[i * 4 + 1] = c(px[i * 3 + 1]); rgba[i * 4 + 2] = c(px[i * 3 + 2])
    }
    guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let img = ctx.makeImage(),
          let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                    UTType.png.identifier as CFString, 1, nil)
    else { print("  (png save failed: \(path))"); return }
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
}

// MARK: - tile grid (replicates RIFEPairTiler.tileGrid geometry — internal there, kept in sync)

struct T { let x, y, tw, th: Int }
func tileGrid(h: Int, w: Int, tileSize: Int, overlap: Int) -> [T] {
    let step = max(tileSize - overlap, 1)
    var tiles: [T] = []
    for ty in stride(from: 0, to: h, by: step) {
        for tx in stride(from: 0, to: w, by: step) {
            let x = min(tx, max(0, w - tileSize)), y = min(ty, max(0, h - tileSize))
            tiles.append(T(x: x, y: y, tw: min(tileSize, w - x), th: min(tileSize, h - y)))
        }
    }
    return tiles
}

/// Per-pixel tile-coverage count. coverage ≥ 2 ⇒ a halo/overlap band = where the seam lives.
func coverageMask(h: Int, w: Int, tiles: [T]) -> [UInt8] {
    var cov = [UInt8](repeating: 0, count: h * w)
    for t in tiles {
        for yy in t.y ..< (t.y + t.th) {
            let row = yy * w
            for xx in t.x ..< (t.x + t.tw) where cov[row + xx] < 255 { cov[row + xx] += 1 }
        }
    }
    return cov
}

// MARK: - error metrics (tiled vs whole-frame reference)

struct Err { let mae: Double; let psnr: Double; let n: Int }
/// MAE + PSNR of `a` vs `ref` restricted to pixels where `mask(pixelIndex)` is true.
func error(_ a: [Float], _ ref: [Float], h: Int, w: Int, keep: (Int) -> Bool) -> Err {
    var se = 0.0, sae = 0.0; var n = 0
    for p in 0 ..< (h * w) where keep(p) {
        for c in 0..<3 {
            let d = Double(a[p * 3 + c] - ref[p * 3 + c])
            se += d * d; sae += abs(d)
        }
        n += 1
    }
    let mse = n > 0 ? se / Double(n * 3) : 0
    let psnr = mse > 0 ? 10 * log10(1.0 / mse) : 99.0   // inputs in [0,1] → MAX_I = 1
    return Err(mae: n > 0 ? sae / Double(n * 3) : 0, psnr: psnr, n: n)
}

// MARK: - run

guard let (aPx, H, W) = loadRGB(aPath), let (bPx, hb, wb) = loadRGB(bPath), hb == H, wb == W else {
    fatalError("failed to load / size-mismatch frames \(aPath) , \(bPath)")
}
print("[EVAL] \(tag): \(W)×\(H)  tileSize=\(tileSize) scale=\(scale)  overlaps=\(overlaps)")

let model = RIFEModel()
try model.loadWeights(from: URL(fileURLWithPath: weightsPath))

let a = MLXArray(aPx, [1, H, W, 3])
let b = MLXArray(bPx, [1, H, W, 3])

// REFERENCE — whole-frame.
let refSampler = PhysSampler(); refSampler.start()
let refArr = model.inference(img0: a, img1: b, timestep: 0.5, scale: scale)
MLX.eval(refArr)
let refPhys = refSampler.stop()
let refPx = refArr.asType(.float32).asArray(Float.self)
savePNG(refPx, h: H, w: W, to: "\(outDir)/\(tag)_ref.png")
print(String(format: "[EVAL] reference whole-frame  peak_phys=%.2f GB", Double(refPhys) / 1e9))
MLX.GPU.clearCache()

struct Row: Codable {
    let tag: String, width: Int, height: Int, tileSize: Int, overlap: Int, scale: Float
    let tiles: Int, seam_mae: Double, seam_psnr: Double, seam_px: Int
    let global_mae: Double, global_psnr: Double, peak_phys_gb: Double, ref_png: String, tiled_png: String
}
var rows: [Row] = []

for ov in overlaps {
    let tiles = tileGrid(h: H, w: W, tileSize: tileSize, overlap: ov)
    let cov = coverageMask(h: H, w: W, tiles: tiles)

    let s = PhysSampler(); s.start()
    let tiler = RIFEPairTiler(tileSize: tileSize, overlap: ov)
    let tiledArr = tiler.interpolate(img0: a, img1: b) { ta, tb in
        let r = model.inference(img0: ta, img1: tb, timestep: 0.5, scale: scale)
        MLX.eval(r)     // eval per tile → the MLX pool frees before the next tile (memory bound)
        return r
    }
    MLX.eval(tiledArr)
    let peak = s.stop()
    let tiledPx = tiledArr.asType(.float32).asArray(Float.self)
    let tiledPNG = "\(outDir)/\(tag)_tiled_ov\(ov).png"
    savePNG(tiledPx, h: H, w: W, to: tiledPNG)

    let seam = error(tiledPx, refPx, h: H, w: W) { cov[$0] >= 2 }
    let glob = error(tiledPx, refPx, h: H, w: W) { _ in true }
    rows.append(Row(tag: tag, width: W, height: H, tileSize: tileSize, overlap: ov, scale: scale,
                    tiles: tiles.count, seam_mae: seam.mae, seam_psnr: seam.psnr, seam_px: seam.n,
                    global_mae: glob.mae, global_psnr: glob.psnr, peak_phys_gb: Double(peak) / 1e9,
                    ref_png: "\(outDir)/\(tag)_ref.png", tiled_png: tiledPNG))
    print(String(format: "[EVAL] ov=%-3d tiles=%-2d  seamPSNR=%.2f dB (mae %.5f, %d px)  globalPSNR=%.2f dB  peak_phys=%.2f GB",
                 ov, tiles.count, seam.psnr, seam.mae, seam.n, glob.psnr, Double(peak) / 1e9))
    MLX.Memory.clearCache()
}

// metrics.json for the orchestrator (ffmpeg adds SSIM/VMAF from the PNGs).
let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
let jsonURL = "\(outDir)/\(tag)_metrics.json"
if let data = try? enc.encode(rows) { try? data.write(to: URL(fileURLWithPath: jsonURL)) }
print("[EVAL] wrote \(jsonURL)  (ref + \(overlaps.count) tiled PNGs in \(outDir))")
