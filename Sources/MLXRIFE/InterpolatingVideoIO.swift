//
//  InterpolatingVideoIO.swift
//  MLXRIFE
//
//  Streaming frame-interpolation transcode: decode frames one at a time, synthesize the
//  intermediate frame(s) for each adjacent pair, encode at factor× the source fps.
//  Adapted from mlx-seedvr2-swift's VideoIO (consolidation into the Layer-2 media service is
//  planned). Forge conventions: HEVC output always BT.709-tagged (#61); bounded memory.
//

import AVFoundation
import CoreVideo
import Foundation
import MLX

public enum RIFEVideoIOError: Error {
    case openFailed(String)
    case noVideoTrack
    case readFailed(String)
    case writeFailed(String)
}

enum InterpolatingVideoIO {

    struct Result {
        let frameRate: Double
        let duration: Double
    }

    /// Interpolate `factor − 1` intermediates between each adjacent frame pair.
    /// `midpoints(prev, next)` returns the synthesized frames at t = k/factor (k = 1…factor−1),
    /// at the same dimensions; cancellation is checked per source frame.
    static func interpolate(
        input: URL,
        output: URL,
        factor: Int,
        midpoints: (CVPixelBuffer, CVPixelBuffer) async throws -> [CVPixelBuffer]
    ) async throws -> Result {
        let asset = AVURLAsset(url: input)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw RIFEVideoIOError.noVideoTrack
        }
        let fps = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration).seconds

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw RIFEVideoIOError.readFailed(reader.error?.localizedDescription ?? "startReading")
        }

        let outFPS = Double(max(fps, 1)) * Double(factor)
        let timescale: CMTimeScale = 60_000
        let outFrameDuration = CMTime(value: CMTimeValue((Double(timescale) / outFPS).rounded()),
                                      timescale: timescale)

        var writer: AVAssetWriter?
        var writerInput: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?
        var outIndex = 0

        func append(_ pb: CVPixelBuffer) async throws {
            if writer == nil {
                let ow = CVPixelBufferGetWidth(pb), oh = CVPixelBufferGetHeight(pb)
                let w = try AVAssetWriter(outputURL: output, fileType: .mp4)
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: ow,
                    AVVideoHeightKey: oh,
                    AVVideoColorPropertiesKey: [
                        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                        AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                    ],
                ])
                input.expectsMediaDataInRealTime = false
                let a = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: input,
                    sourcePixelBufferAttributes: [
                        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                        kCVPixelBufferWidthKey as String: ow,
                        kCVPixelBufferHeightKey as String: oh,
                    ])
                w.add(input)
                guard w.startWriting() else {
                    throw RIFEVideoIOError.writeFailed(w.error?.localizedDescription ?? "startWriting")
                }
                w.startSession(atSourceTime: .zero)
                writer = w; writerInput = input; adaptor = a
            }
            guard let inp = writerInput, let adaptor else { return }
            while !inp.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 2_000_000)
                try Task.checkCancellation()
            }
            let t = CMTimeMultiply(outFrameDuration, multiplier: Int32(outIndex))
            guard adaptor.append(pb, withPresentationTime: t) else {
                throw RIFEVideoIOError.writeFailed(writer?.error?.localizedDescription ?? "append \(outIndex)")
            }
            outIndex += 1
        }

        var prev: CVPixelBuffer? = nil
        while let sample = readerOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let frame = CMSampleBufferGetImageBuffer(sample) else { continue }
            if let p = prev {
                try await append(p)
                for mid in try await midpoints(p, frame) {
                    try await append(mid)
                }
            }
            prev = frame
        }
        if let last = prev { try await append(last) }

        if reader.status == .failed {
            throw RIFEVideoIOError.readFailed(reader.error?.localizedDescription ?? "reader failed")
        }
        guard let writer, let writerInput else {
            throw RIFEVideoIOError.readFailed("no frames decoded")
        }
        writerInput.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed {
            throw RIFEVideoIOError.writeFailed(writer.error?.localizedDescription ?? "finishWriting")
        }
        return Result(frameRate: outFPS, duration: duration)
    }
}
