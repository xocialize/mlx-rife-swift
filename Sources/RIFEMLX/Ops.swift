//
//  Ops.swift
//  RIFEMLX
//
//  Hand-rolled NHWC spatial ops — isomorphic to rife-mlx's ops/{grid_sample,interpolate,spatial}.py
//  (each parity-locked vs torch there). The crux surface of the port.
//

import Foundation
import MLX

/// Bilinear grid_sample (NHWC) — torch.nn.functional.grid_sample equivalent.
///
/// Conventions (match torch): grid[..., 0] = x (width, normalized [-1,1]), grid[..., 1] = y.
/// align_corners=true: -1 → pixel 0, +1 → pixel (size-1). padding_mode=border: neighbor integer
/// indices clamp to the valid range; bilinear weights come from the UNCLAMPED coordinate.
///
/// - Parameters:
///   - input: `[N, H, W, C]`
///   - grid: `[N, gH, gW, 2]` in [-1, 1]
/// - Returns: `[N, gH, gW, C]`
public func gridSampleBilinear(_ input: MLXArray, grid: MLXArray,
                               alignCorners: Bool = true) -> MLXArray {
    let N = input.shape[0], H = input.shape[1], W = input.shape[2], C = input.shape[3]
    let gH = grid.shape[1], gW = grid.shape[2]

    let gx = grid[0..., 0..., 0..., 0]
    let gy = grid[0..., 0..., 0..., 1]
    let ix: MLXArray
    let iy: MLXArray
    if alignCorners {
        ix = (gx + 1) * 0.5 * Float(W - 1)
        iy = (gy + 1) * 0.5 * Float(H - 1)
    } else {
        ix = ((gx + 1) * Float(W) - 1) * 0.5
        iy = ((gy + 1) * Float(H) - 1) * 0.5
    }

    let x0 = floor(ix), y0 = floor(iy)
    let x1 = x0 + 1, y1 = y0 + 1
    let wx1 = ix - x0, wx0 = 1.0 - (ix - x0)
    let wy1 = iy - y0, wy0 = 1.0 - (iy - y0)

    func clampX(_ a: MLXArray) -> MLXArray { clip(a, min: 0, max: Float(W - 1)).asType(.int32) }
    func clampY(_ a: MLXArray) -> MLXArray { clip(a, min: 0, max: Float(H - 1)).asType(.int32) }
    let x0c = clampX(x0), x1c = clampX(x1)
    let y0c = clampY(y0), y1c = clampY(y1)

    let inputFlat = input.reshaped([N, H * W, C])

    func gather(_ yc: MLXArray, _ xc: MLXArray) -> MLXArray {
        var idx = (yc * Int32(W) + xc).reshaped([N, gH * gW, 1])
        idx = broadcast(idx, to: [N, gH * gW, C])
        return takeAlong(inputFlat, idx, axis: 1).reshaped([N, gH, gW, C])
    }

    let v00 = gather(y0c, x0c), v01 = gather(y0c, x1c)
    let v10 = gather(y1c, x0c), v11 = gather(y1c, x1c)

    let w00 = (wy0 * wx0).expandedDimensions(axis: -1)
    let w01 = (wy0 * wx1).expandedDimensions(axis: -1)
    let w10 = (wy1 * wx0).expandedDimensions(axis: -1)
    let w11 = (wy1 * wx1).expandedDimensions(axis: -1)
    return v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}

/// Sample source coordinates for one resampled axis (torch F.interpolate semantics).
private func sampleCoords(out: Int, inSize: Int, alignCorners: Bool) -> MLXArray {
    let dst = MLXArray(Array(0..<out).map { Float($0) })
    if alignCorners {
        let scale = out > 1 ? Float(inSize - 1) / Float(out - 1) : 0
        return dst * scale
    }
    let scale = Float(inSize) / Float(out)
    return (dst + 0.5) * scale - 0.5
}

/// Resample one spatial axis with bilinear weights + border clamp.
private func bilinear1D(_ x: MLXArray, axis: Int, out: Int, alignCorners: Bool) -> MLXArray {
    let inSize = x.shape[axis]
    if inSize == out { return x }
    let src = sampleCoords(out: out, inSize: inSize, alignCorners: alignCorners)
    let i0f = floor(src)
    let w1 = src - i0f, w0 = 1.0 - (src - i0f)
    let i0 = clip(i0f, min: 0, max: Float(inSize - 1)).asType(.int32)
    let i1 = clip(i0f + 1, min: 0, max: Float(inSize - 1)).asType(.int32)
    let g0 = take(x, i0, axis: axis)
    let g1 = take(x, i1, axis: axis)
    var shape = [Int](repeating: 1, count: x.ndim)
    shape[axis] = out
    return g0 * w0.reshaped(shape) + g1 * w1.reshaped(shape)
}

/// Bilinear resize (NHWC) — F.interpolate(mode: "bilinear") equivalent.
/// - Parameter x: `[N, H, W, C]`; resizes H and W.
public func interpolateBilinear(_ x: MLXArray, scaleFactor: Float,
                                alignCorners: Bool = false) -> MLXArray {
    let H = x.shape[1], W = x.shape[2]
    let oH = Int((Float(H) * scaleFactor).rounded())
    let oW = Int((Float(W) * scaleFactor).rounded())
    var out = bilinear1D(x, axis: 1, out: oH, alignCorners: alignCorners)
    out = bilinear1D(out, axis: 2, out: oW, alignCorners: alignCorners)
    return out
}

/// NHWC pixel-shuffle with the torch-matching (C, r, r) channel split (mlx-porting pitfall #7).
/// `[N, H, W, C*r²] → [N, H*r, W*r, C]`.
public func pixelShuffleNHWC(_ x: MLXArray, _ r: Int) -> MLXArray {
    let n = x.shape[0], h = x.shape[1], w = x.shape[2], cIn = x.shape[3]
    let c = cIn / (r * r)
    var out = x.reshaped([n, h, w, c, r, r])
    out = out.transposed(0, 1, 4, 2, 5, 3)  // (N, H, r_i, W, r_j, C)
    return out.reshaped([n, h * r, w * r, c])
}

/// Backward warp via grid_sample — isomorphic to rife-mlx's warplayer.py.
/// - Parameters:
///   - input: `[N, H, W, C]`
///   - flow: `[N, H, W, 2]` pixel-space flow (channel 0 = x, 1 = y)
public func warp(_ input: MLXArray, flow: MLXArray) -> MLXArray {
    let N = flow.shape[0], H = flow.shape[1], W = flow.shape[2]
    let xs = broadcast(linspace(Float(-1), Float(1), count: W).reshaped([1, 1, W, 1]),
                       to: [N, H, W, 1])
    let ys = broadcast(linspace(Float(-1), Float(1), count: H).reshaped([1, H, 1, 1]),
                       to: [N, H, W, 1])
    let grid = concatenated([xs, ys], axis: -1)
    let fx = flow[0..., 0..., 0..., 0..<1] / (Float(W - 1) / 2.0)
    let fy = flow[0..., 0..., 0..., 1..<2] / (Float(H - 1) / 2.0)
    let g = grid + concatenated([fx, fy], axis: -1)
    return gridSampleBilinear(input, grid: g, alignCorners: true)
}
