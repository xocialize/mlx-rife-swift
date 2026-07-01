import Foundation
import CoreVideo
import FrameStreamNative
import MLXToolKit
import MLXProfiling
import MLX
import Hub
import RIFEMLX

/// Errors at the RIFE package boundary.
public enum RIFEPackageError: Error, Equatable {
    case unsupportedFactor(Int)
    case frameConversionFailed
}

/// An MLXEngine `frameInterpolate` package over **Practical-RIFE 4.25** — raises a video's frame
/// rate by synthesizing intermediates (factor 2 = one midpoint per adjacent pair at t=0.5;
/// factor N = N−1 frames at t=k/N). Chains onto low-fps T2V output or any decoded source.
///
/// A thin conformance wrapper over the parity-locked `RIFEMLX` core (rife-mlx-swift; cosine 1.0
/// vs the Python reference on real weights). Frames stream decode → interpolate → encode
/// (HEVC, BT.709-tagged) with per-source-frame cancellation.
@InferenceActor
public final class RIFEInterpolatePackage: ModelPackage {
    public typealias Configuration = RIFEConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Practical-RIFE (hzwer) is MIT; the mlx-community weights + port code are MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/RIFE-4.25", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split (born-clean): ~21 MB fp32 weights resident (declare 150 MB with overhead).
                // The per-pair pyramid activation is the transient and is RESOLUTION-LINEAR (RIFE has
                // no tiling — the whole frame is resident through the pyramid). RE-BASELINED 2026-07-01
                // against REAL phys_footprint (HostMemory) on the video path, factor 2: resident floor
                // 0.04 GB; peak activation 0.77 GB @360p, 2.37 GB @720p, 3.86 GB @1080p. Declared for a
                // 1080p max input (4.0 GB). Factor-independent (each intermediate evals separately).
                // NOTE: 4K would extrapolate to ~15 GB and exceed most Macs' GPU working set — RIFE
                // needs a tiling pass (like SeedVR2's MLXTileProcessor) before 4K is admissible; tracked
                // as a separate enhancement. Old declared 1.5 GB was safe only to ~540p.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 150_000_000,
                                            peakActivationBytes: 4_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                FrameInterpolateContract.descriptor(
                    name: "rife-interpolate",
                    summary: "RIFE 4.25 frame interpolation: 2x/4x frame-rate up-conversion via synthesized intermediates."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: RIFEModel?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
        let dir = try await hub.snapshot(from: Hub.Repo(id: configuration.repo),
                                         matching: ["model.safetensors"]) { progress, speed in
            WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
        }
        let m = RIFEModel()
        try m.loadWeights(from: dir.appendingPathComponent("model.safetensors"))
        model = m
    }

    public func unload() async {
        model = nil
        // Dropping the ref alone leaves weight/activation buffers in MLX's pool, so phys_footprint
        // doesn't fall and engine.evict / R-MEM-1 can't reclaim — flush the pool.
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .frameInterpolate,
              let req = request as? FrameInterpolateRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let factor = req.factor ?? configuration.defaultFactor
        guard factor >= 2, factor <= 8 else { throw RIFEPackageError.unsupportedFactor(factor) }
        try Task.checkCancellation()

        let tmpDir = FileManager.default.temporaryDirectory
        let inURL = tmpDir.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(req.video.format.rawValue)
        let outURL = tmpDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        try req.video.data.write(to: inURL)
        defer {
            try? FileManager.default.removeItem(at: inURL)
            try? FileManager.default.removeItem(at: outURL)
        }

        let scale = configuration.scale

        // Native AVFoundation media seam (frame-stream-native): decode → pairwise 1:factor
        // insertion → HEVC/BT.709 at factor× the source fps. Input is a native container
        // (mp4/mov/m4v) — non-native sources are normalized upstream by format-bridge. The probe
        // runs up front because uniform re-timing needs the target fps before the stream starts.
        let info = try await NativeFrameStream.probe(url: inURL)
        let outFPS = max(info.frameRate, 1) * Double(factor)

        final class Window: @unchecked Sendable { var prev: CVPixelBuffer?; var pair = 0 }
        let win = Window()

        // Per-intermediate MLX inference is profiled (MLX_PROFILE=1). RIFE's activation is
        // resolution-linear (no tiling) — the region's phys/⚠PAGING readings flag when a high-res
        // input approaches the Metal working-set ceiling (the pre-4K-tiling admission signal).
        let prof = MLXProfiler.shared
        prof.beginRun("rife frameInterpolate factor=\(factor)")
        let result = try await NativeFrameStream.run(
            input: inURL, output: outURL, timing: .uniform(fps: outFPS),
            transform: { frame in
                try Task.checkCancellation()
                defer { win.prev = frame; win.pair += 1 }
                guard let p = win.prev else { return [] }   // prime the pairwise window
                guard let a = Self.rgbNHWC(p), let b = Self.rgbNHWC(frame) else {
                    throw RIFEPackageError.frameConversionFailed
                }
                var outs: [CVPixelBuffer] = [p]
                for k in 1..<factor {
                    let t = Float(k) / Float(factor)
                    let mid = prof.region("interp", "mid", index: win.pair, note: "t=\(t)") { () -> MLXArray in
                        let m = model.inference(img0: a, img1: b, timestep: t, scale: scale)
                        MLX.eval(m)   // eval INSIDE the region so the lazy compute is timed honestly
                        return m
                    }
                    guard let pb = Self.pixelBuffer(fromRGBNHWC: mid,
                                                    width: mid.shape[2], height: mid.shape[1]) else {
                        throw RIFEPackageError.frameConversionFailed
                    }
                    outs.append(pb)
                }
                return outs
            },
            flush: { win.prev.map { [$0] } ?? [] }
        )
        prof.endRun(denominators: ["pair": Double(max(win.pair, 1))])

        let data = try Data(contentsOf: outURL)
        return FrameInterpolateResponse(
            video: Video(format: .mp4, data: data,
                         durationSeconds: result.sourceDuration, frameRate: outFPS),
            appliedFactor: factor)
    }

    // MARK: - Pixel conversion (shared-convention utilities; Layer-2 consolidation planned)

    /// BGRA `CVPixelBuffer` → `[1, H, W, 3]` RGB float NHWC in [0,1].
    nonisolated static func rgbNHWC(_ bgra: CVPixelBuffer) -> MLXArray? {
        CVPixelBufferLockBaseAddress(bgra, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(bgra, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(bgra) else { return nil }
        let width = CVPixelBufferGetWidth(bgra), height = CVPixelBufferGetHeight(bgra)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(bgra)
        let src = base.assumingMemoryBound(to: UInt8.self)

        var rgb = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            let row = y * bytesPerRow
            let drow = y * width * 3
            for x in 0..<width {
                let s = row + x * 4
                let d = drow + x * 3
                rgb[d + 0] = Float(src[s + 2]) / 255.0
                rgb[d + 1] = Float(src[s + 1]) / 255.0
                rgb[d + 2] = Float(src[s + 0]) / 255.0
            }
        }
        return MLXArray(rgb, [1, height, width, 3])
    }

    /// `[1, H, W, 3]` RGB float (clamped) → BGRA `CVPixelBuffer`.
    nonisolated static func pixelBuffer(fromRGBNHWC array: MLXArray, width: Int, height: Int) -> CVPixelBuffer? {
        let rgb = array.asType(.float32).asArray(Float.self)
        guard rgb.count >= width * height * 3 else { return nil }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let dstBase = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let dst = dstBase.assumingMemoryBound(to: UInt8.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        @inline(__always) func clamp8(_ v: Float) -> UInt8 { UInt8(max(0, min(255, v * 255))) }
        for y in 0..<height {
            let row = y * bytesPerRow
            let srow = y * width * 3
            for x in 0..<width {
                let d = row + x * 4
                let s = srow + x * 3
                dst[d + 0] = clamp8(rgb[s + 2])
                dst[d + 1] = clamp8(rgb[s + 1])
                dst[d + 2] = clamp8(rgb[s + 0])
                dst[d + 3] = 255
            }
        }
        return buffer
    }
}

extension RIFEInterpolatePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RIFEInterpolatePackage.self)
    }
}
