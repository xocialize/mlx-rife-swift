import Foundation
import CoreVideo
import MLXToolKit
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
                // ~21 MB fp32 weights; the working set is the per-pair pyramid activations.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 1_500_000_000)],
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
                                         matching: ["model.safetensors"])
        let m = RIFEModel()
        try m.loadWeights(from: dir.appendingPathComponent("model.safetensors"))
        model = m
    }

    public func unload() async {
        model = nil
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
        let result = try await InterpolatingVideoIO.interpolate(
            input: inURL, output: outURL, factor: factor
        ) { prev, next in
            try Task.checkCancellation()
            guard let a = Self.rgbNHWC(prev), let b = Self.rgbNHWC(next) else {
                throw RIFEPackageError.frameConversionFailed
            }
            var mids: [CVPixelBuffer] = []
            for k in 1..<factor {
                let t = Float(k) / Float(factor)
                let mid = model.inference(img0: a, img1: b, timestep: t, scale: scale)
                MLX.eval(mid)
                guard let pb = Self.pixelBuffer(fromRGBNHWC: mid,
                                                width: mid.shape[2], height: mid.shape[1]) else {
                    throw RIFEPackageError.frameConversionFailed
                }
                mids.append(pb)
            }
            return mids
        }

        let data = try Data(contentsOf: outURL)
        return FrameInterpolateResponse(
            video: Video(format: .mp4, data: data,
                         durationSeconds: result.duration, frameRate: result.frameRate),
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
