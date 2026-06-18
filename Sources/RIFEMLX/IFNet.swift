//
//  IFNet.swift
//  RIFEMLX
//
//  IFNet (RIFE 4.25) — isomorphic to rife-mlx's model/IFNet_HDv3.py (itself line-isomorphic
//  to the pinned upstream train_log/IFNet_HDv3.py). NHWC throughout.
//
//  Arch (pinned): 5 IFBlocks c=[192,128,96,64,32], in_planes=[15,28,28,28,28];
//  conv = Conv2d + LeakyReLU(0.2); ResConv = conv*beta + x → LeakyReLU; lastconv =
//  ConvTranspose2d(c, 52, 4, 2, 1) + PixelShuffle(2) → 13ch (flow4 + mask1 + feat8);
//  Head encoder → 4ch; interpolate align_corners=false; warp align_corners=true.
//
//  Weight keys match the mlx-community/RIFE-4.25 checkpoint (post-conversion MLX keys):
//  conv0.{j}.conv.*, convblock.{j}.conv.* / .beta, lastconv.*, encode.cnn{0-3}.*.
//

import Foundation
import MLX
import MLXNN

private let slope: Float = 0.2

@inline(__always) private func lrelu(_ x: MLXArray) -> MLXArray {
    leakyRelu(x, negativeSlope: slope)
}

/// conv = Conv2d + LeakyReLU(0.2). Key: `<p>.conv.weight/bias`.
final class ConvBlock: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d

    init(_ i: Int, _ o: Int, kernel: Int = 3, stride: Int = 1, padding: Int = 1) {
        self._conv.wrappedValue = Conv2d(inputChannels: i, outputChannels: o,
                                         kernelSize: IntOrPair(kernel),
                                         stride: IntOrPair(stride),
                                         padding: IntOrPair(padding))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lrelu(conv(x))
    }
}

/// conv*beta + x → LeakyReLU. Keys: `.conv.weight/bias`, `.beta`.
final class ResConv: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d
    @ParameterInfo var beta: MLXArray

    init(_ c: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: c, outputChannels: c,
                                         kernelSize: 3, stride: 1, padding: 1)
        self._beta.wrappedValue = MLXArray.ones([1, 1, 1, c])  // NHWC per-channel
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lrelu(conv(x) * beta + x)
    }
}

/// Feature encoder. Keys: cnn0, cnn1, cnn2 (Conv2d), cnn3 (ConvTranspose2d).
final class Head: Module, UnaryLayer {
    @ModuleInfo var cnn0: Conv2d
    @ModuleInfo var cnn1: Conv2d
    @ModuleInfo var cnn2: Conv2d
    @ModuleInfo var cnn3: ConvTransposed2d

    override init() {
        self._cnn0.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 16, kernelSize: 3, stride: 2, padding: 1)
        self._cnn1.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 3, stride: 1, padding: 1)
        self._cnn2.wrappedValue = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 3, stride: 1, padding: 1)
        self._cnn3.wrappedValue = ConvTransposed2d(inputChannels: 16, outputChannels: 4, kernelSize: 4, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = lrelu(cnn0(x))
        h = lrelu(cnn1(h))
        h = lrelu(cnn2(h))
        return cnn3(h)
    }
}

final class IFBlock: Module {
    @ModuleInfo var conv0: [ConvBlock]
    @ModuleInfo var convblock: [ResConv]
    @ModuleInfo var lastconv: ConvTransposed2d

    init(inPlanes: Int, c: Int) {
        self._conv0.wrappedValue = [
            ConvBlock(inPlanes, c / 2, kernel: 3, stride: 2, padding: 1),
            ConvBlock(c / 2, c, kernel: 3, stride: 2, padding: 1),
        ]
        self._convblock.wrappedValue = (0..<8).map { _ in ResConv(c) }
        self._lastconv.wrappedValue = ConvTransposed2d(inputChannels: c, outputChannels: 4 * 13,
                                                       kernelSize: 4, stride: 2, padding: 1)
    }

    /// Returns (flow [N,h,w,4], mask [N,h,w,1], feat [N,h,w,8]).
    func callAsFunction(_ xIn: MLXArray, flow flowIn: MLXArray?, scale: Float) -> (MLXArray, MLXArray, MLXArray) {
        var x = interpolateBilinear(xIn, scaleFactor: 1.0 / scale, alignCorners: false)
        if let flowIn {
            let f = interpolateBilinear(flowIn, scaleFactor: 1.0 / scale, alignCorners: false) * (1.0 / scale)
            x = concatenated([x, f], axis: -1)
        }
        var feat = conv0[0](x)
        feat = conv0[1](feat)
        for blk in convblock {
            feat = blk(feat)
        }
        var tmp = pixelShuffleNHWC(lastconv(feat), 2)
        tmp = interpolateBilinear(tmp, scaleFactor: scale, alignCorners: false)
        let flow = tmp[0..., 0..., 0..., ..<4] * scale
        let mask = tmp[0..., 0..., 0..., 4..<5]
        let outFeat = tmp[0..., 0..., 0..., 5...]
        return (flow, mask, outFeat)
    }
}

public final class IFNet: Module {
    @ModuleInfo var block0: IFBlock
    @ModuleInfo var block1: IFBlock
    @ModuleInfo var block2: IFBlock
    @ModuleInfo var block3: IFBlock
    @ModuleInfo var block4: IFBlock
    @ModuleInfo var encode: Head

    public override init() {
        self._block0.wrappedValue = IFBlock(inPlanes: 7 + 8, c: 192)
        self._block1.wrappedValue = IFBlock(inPlanes: 8 + 4 + 8 + 8, c: 128)
        self._block2.wrappedValue = IFBlock(inPlanes: 8 + 4 + 8 + 8, c: 96)
        self._block3.wrappedValue = IFBlock(inPlanes: 8 + 4 + 8 + 8, c: 64)
        self._block4.wrappedValue = IFBlock(inPlanes: 8 + 4 + 8 + 8, c: 32)
        self._encode.wrappedValue = Head()
    }

    /// - Parameters:
    ///   - x: `[N, H, W, 6]` (img0 | img1 on the channel axis), values in [0, 1].
    ///   - timestep: interpolation position in (0, 1).
    ///   - scaleList: coarse-to-fine pyramid (already divided by the user `scale` knob).
    /// - Returns: the merged middle frame `[N, H, W, 3]`.
    public func callAsFunction(_ x: MLXArray, timestep: Float, scaleList: [Float]) -> MLXArray {
        let channel = x.shape[3] / 2
        let img0 = x[0..., 0..., 0..., ..<channel]
        let img1 = x[0..., 0..., 0..., channel...]
        let N = img0.shape[0], H = img0.shape[1], W = img0.shape[2]
        let t = MLXArray.full([N, H, W, 1], values: MLXArray(timestep))

        let f0 = encode(img0[0..., 0..., 0..., ..<3])
        let f1 = encode(img1[0..., 0..., 0..., ..<3])

        var warpedImg0 = img0
        var warpedImg1 = img1
        var flow: MLXArray? = nil
        var mask: MLXArray! = nil
        var feat: MLXArray! = nil
        let blocks = [block0, block1, block2, block3, block4]

        for i in 0..<5 {
            if flow == nil {
                let (f, m, ft) = blocks[i](
                    concatenated([img0[0..., 0..., 0..., ..<3], img1[0..., 0..., 0..., ..<3],
                                  f0, f1, t], axis: -1),
                    flow: nil, scale: scaleList[i])
                flow = f; mask = m; feat = ft
            } else {
                let wf0 = warp(f0, flow: flow![0..., 0..., 0..., ..<2])
                let wf1 = warp(f1, flow: flow![0..., 0..., 0..., 2..<4])
                let (fd, m0, ft) = blocks[i](
                    concatenated([warpedImg0[0..., 0..., 0..., ..<3], warpedImg1[0..., 0..., 0..., ..<3],
                                  wf0, wf1, t, mask, feat], axis: -1),
                    flow: flow, scale: scaleList[i])
                mask = m0; feat = ft
                flow = flow! + fd
            }
            warpedImg0 = warp(img0, flow: flow![0..., 0..., 0..., ..<2])
            warpedImg1 = warp(img1, flow: flow![0..., 0..., 0..., 2..<4])
        }
        let m = sigmoid(mask)
        return warpedImg0 * m + warpedImg1 * (1 - m)
    }
}
