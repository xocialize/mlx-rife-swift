// swift-tools-version: 6.2
import PackageDescription

// mlx-rife-swift — the MLXEngine `frameInterpolate` package over Practical-RIFE 4.25.
// A Video→Video transform of the visual optimization tier: raises frame rate by synthesizing
// intermediate frames (factor 2 = one midpoint per adjacent pair; 4 = three at t=k/4).
// Thin conformance layer over the parity-locked rife-mlx-swift core. Module is `MLXRIFE`.
let package = Package(
    name: "mlx-rife-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXRIFE", targets: ["MLXRIFE"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(path: "../format-bridge"),
        .package(url: "https://github.com/xocialize/rife-mlx-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "MLXRIFE",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FormatBridge", package: "format-bridge"),
                .product(name: "RIFEMLX", package: "rife-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXRIFETests",
            dependencies: [
                "MLXRIFE",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FormatBridge", package: "format-bridge"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
