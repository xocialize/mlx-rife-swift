# mlx-rife-swift

The MLXEngine **`frameInterpolate`** package over [Practical-RIFE 4.25](https://github.com/xocialize/rife-mlx-swift) — frame-rate up-conversion on Apple Silicon.

A `Video → Video` transform of the visual optimization tier: synthesizes intermediate frames
(factor 2 = one midpoint per adjacent pair at t=0.5; factor N = N−1 at t=k/N) and re-encodes at
factor× the source fps (HEVC, BT.709-tagged). Chains onto low-fps T2V output or any decoded
source. The core is **parity-locked** (cosine 1.0 vs the Python `rife-mlx` reference on real
weights); frames stream decode → interpolate → encode with per-frame cancellation.

## Weights

[`mlx-community/RIFE-4.25`](https://huggingface.co/mlx-community/RIFE-4.25) (MIT, ~21 MB) —
downloaded into the engine's model store on first load.

## Usage

```swift
import MLXServeCore
import MLXRIFE

let engine = MLXServeEngine()
try await engine.register(RIFEInterpolatePackage.registration, configuration: RIFEConfiguration())

let resp = try await engine.run(FrameInterpolateRequest(video: clip, factor: 2)) as! FrameInterpolateResponse
// resp.video — 2× fps HEVC .mp4; resp.appliedFactor == 2
```

Requirements: macOS 26+ (Apple Silicon, Metal GPU). MIT throughout.
