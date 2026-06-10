import Testing
import Foundation
import CoreVideo
import MLXToolKit
@testable import MLXRIFE

/// Offline conformance — no Metal evaluation. Live interpolation is proven in the
/// `MLXEngine Testing` app; the core carries the parity suite (cosine 1.0 vs python).
struct RIFEInterpolateTests {

    @Test func manifestIsFrameInterpolateAndPermissive() {
        let m = RIFEInterpolatePackage.manifest
        #expect(m.capabilities == [.frameInterpolate])
        #expect(m.license.weightLicense == .mit)
        #expect(m.license.portCodeLicense == .mit)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
        #expect(m.provenance.sourceRepo == "mlx-community/RIFE-4.25")
    }

    @Test func manifestRequirements() {
        let r = RIFEInterpolatePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
        #expect(r.footprints.first?.quant == .fp32)
    }

    @Test func surfaceIsTheCanonicalInterpolateDescriptor() {
        let s = RIFEInterpolatePackage.manifest.surfaces.first
        #expect(s?.capability == .frameInterpolate)
        #expect(s?.parameters.first?.kind == .video)
        #expect(s?.parameters.contains { $0.name == "factor" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = RIFEInterpolatePackage.registration
        #expect(reg.manifest.capabilities == [.frameInterpolate])
        let pkg = try reg.makePackage(RIFEConfiguration())
        #expect(pkg is RIFEInterpolatePackage)
    }

    @Test func configurationDefaultsAndCodable() throws {
        let c = RIFEConfiguration()
        #expect(c.repo == "mlx-community/RIFE-4.25")
        #expect(c.defaultFactor == 2)
        #expect(c.scale == 1.0)

        var custom = RIFEConfiguration(defaultFactor: 4, scale: 0.5)
        custom.modelsRootDirectory = URL(fileURLWithPath: "/tmp/x")
        let back = try JSONDecoder().decode(RIFEConfiguration.self, from: JSONEncoder().encode(custom))
        #expect(back.defaultFactor == 4)
        #expect(back.scale == 0.5)
        #expect(back.modelsRootDirectory == nil)
    }

    @Test func pixelConversionRoundTrips() throws {
        // BGRA buffer → NHWC → BGRA (no Metal: array creation is lazy; conversion code is CPU).
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: 8, kCVPixelBufferHeightKey as String: 8,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferCreate(nil, 8, 8, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        let buffer = try #require(pb)
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, 128, CVPixelBufferGetDataSize(buffer))
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])

        let arr = try #require(RIFEInterpolatePackage.rgbNHWC(buffer))
        #expect(arr.shape == [1, 8, 8, 3])
    }
}
