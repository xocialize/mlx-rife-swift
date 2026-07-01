// swift-tools-version: 6.2
import PackageDescription

// mlx-rife-swift — Practical-RIFE 4.25 frame interpolation for MLXEngine. ONE repo, TWO products:
//   • RIFEMLX — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXRIFE — the MLXEngine `frameInterpolate` ModelPackage over that core
// Consolidated 2026-06-18: the former standalone `rife-mlx-swift` core was folded in (archived).
let package = Package(
    name: "mlx-rife-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "RIFEMLX", targets: ["RIFEMLX"]),
        .library(name: "MLXRIFE", targets: ["MLXRIFE"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.17.0"),
        .package(url: "https://github.com/xocialize/frame-stream-native.git", from: "0.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "RIFEMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXRIFE",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                "RIFEMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "RIFEMLXTests",
            dependencies: [
                "RIFEMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXRIFETests",
            dependencies: [
                "MLXRIFE",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
