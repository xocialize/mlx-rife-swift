import Foundation
import MLXToolKit

/// Init-time configuration for `RIFEInterpolatePackage` (C9).
public struct RIFEConfiguration: PackageConfiguration, ModelStorable {
    /// HF repo holding the RIFE 4.25 MLX weights.
    public var repo: String
    /// Default frame-rate multiplier when a request doesn't specify one.
    public var defaultFactor: Int
    /// Pyramid scale knob (1.0 default; 0.5 for ≥4K inputs). With tiling, keep 1.0 for
    /// native-resolution flow — tiling bounds memory instead. `< 1` remains the escape hatch
    /// for pathological cross-seam motion (see docs/tiled-interpolation-design.md §5).
    public var scale: Float
    /// Above this input pixel count the interpolate path switches from a whole-frame forward to
    /// the memory-bounded `RIFEPairTiler`. Default = 1920·1080 (2_073_600): the ≤1080p
    /// whole-frame path is byte-for-byte unchanged; tiling only engages above it. `0` disables
    /// tiling (always whole-frame).
    public var tileThresholdPixels: Int
    /// Tile side length (including the motion context halo) when tiling engages. Choose so a
    /// tile's pixel count stays within the activation budget — default 1024 (≈1.05 Mpx,
    /// ~1.95 GB, under the 3.86 GB measured at 1080p).
    public var tileSize: Int
    /// Tile overlap / motion-halo width in pixels. Must cover expected per-frame motion near
    /// tile boundaries; default 128 (see design doc §4).
    public var tileOverlap: Int
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/RIFE-4.25",
                defaultFactor: Int = 2,
                scale: Float = 1.0,
                tileThresholdPixels: Int = 1920 * 1080,
                tileSize: Int = 1024,
                tileOverlap: Int = 128,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.defaultFactor = defaultFactor
        self.scale = scale
        self.tileThresholdPixels = tileThresholdPixels
        self.tileSize = tileSize
        self.tileOverlap = tileOverlap
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, defaultFactor, scale, tileThresholdPixels, tileSize, tileOverlap
    }
}
