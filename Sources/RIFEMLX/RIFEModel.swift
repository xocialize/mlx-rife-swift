//
//  RIFEModel.swift
//  RIFEMLX
//
//  Model wrapper — isomorphic to rife-mlx's model/RIFE_HDv3.py: wraps IFNet, pads H/W to a
//  multiple of pad_to (64; the 5-block downsample factor — scaled by 1/scale so the coarsest
//  interpolation round-trips), runs the pyramid, crops back.
//

import Foundation
import MLX
import MLXNN

public enum RIFEError: Error, CustomStringConvertible {
    case weightsFileNotFound(String)
    case loadFailed(String)

    public var description: String {
        switch self {
        case .weightsFileNotFound(let p): return "RIFE weights not found: \(p)"
        case .loadFailed(let d): return "RIFE weight load failed: \(d)"
        }
    }
}

public final class RIFEModel: @unchecked Sendable {
    public let flownet: IFNet
    public let scaleList: [Float]
    public let padTo: Int

    public init(scaleList: [Float] = [16, 8, 4, 2, 1], padTo: Int = 64) {
        self.flownet = IFNet()
        self.scaleList = scaleList
        self.padTo = padTo
    }

    /// Load the mlx-community/RIFE-4.25 checkpoint (`model.safetensors`).
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw RIFEError.weightsFileNotFound(url.path)
        }
        do {
            let arrays = try MLX.loadArrays(url: url)
            let parameters = ModuleParameters.unflattened(arrays)
            try flownet.update(parameters: parameters, verify: .noUnusedKeys)
            MLX.eval(flownet.parameters())
        } catch let e as RIFEError {
            throw e
        } catch {
            throw RIFEError.loadFailed(error.localizedDescription)
        }
    }

    /// Interpolate the frame at `timestep` between two frames.
    /// - Parameters:
    ///   - img0, img1: `[N, H, W, 3]` RGB in [0, 1].
    ///   - timestep: position in (0, 1); 0.5 = midpoint.
    ///   - scale: pyramid scale knob (1.0 default; <1 for very high-res inputs).
    /// - Returns: `[N, H, W, 3]` the interpolated frame.
    public func inference(img0: MLXArray, img1: MLXArray,
                          timestep: Float = 0.5, scale: Float = 1.0) -> MLXArray {
        let H = img0.shape[1], W = img0.shape[2]
        // Pad must grow with 1/scale so the coarsest-scale interpolation round-trips exactly.
        let pad = max(padTo, Int((Float(padTo) / scale).rounded()))
        let ph = ((H - 1) / pad + 1) * pad
        let pw = ((W - 1) / pad + 1) * pad
        let widths = [IntOrPair((0, 0)), IntOrPair((0, ph - H)), IntOrPair((0, pw - W)), IntOrPair((0, 0))]
        let i0 = padded(img0, widths: widths)
        let i1 = padded(img1, widths: widths)

        let scales = scaleList.map { $0 / scale }
        let x = concatenated([i0, i1], axis: -1)
        let merged = flownet(x, timestep: timestep, scaleList: scales)
        return merged[0..., ..<H, ..<W, 0...]
    }
}
