import Foundation
import MLXToolKit

/// Init-time configuration for `RIFEInterpolatePackage` (C9).
public struct RIFEConfiguration: PackageConfiguration, ModelStorable {
    /// HF repo holding the RIFE 4.25 MLX weights.
    public var repo: String
    /// Default frame-rate multiplier when a request doesn't specify one.
    public var defaultFactor: Int
    /// Pyramid scale knob (1.0 default; 0.5 for ≥4K inputs).
    public var scale: Float
    /// Where weights are materialized. Set by the engine from its `ModelStore`; `nil` → the
    /// default swift-transformers cache. Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(repo: String = "mlx-community/RIFE-4.25",
                defaultFactor: Int = 2,
                scale: Float = 1.0,
                modelsRootDirectory: URL? = nil) {
        self.repo = repo
        self.defaultFactor = defaultFactor
        self.scale = scale
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, defaultFactor, scale
    }
}
