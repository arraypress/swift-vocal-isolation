//
//  MDXSeparatorTests.swift
//  VocalIsolationTests
//
//  Created by David Sherlock on 2026.
//
//  The Core AI MDX23C against the same Python reference the Core ML package is
//  held to, plus the chunk layout both share. The model-backed test is gated on
//  the asset installed where `stems` puts it (or $STEMS_MDX_ASSET) and skips
//  when absent, as on CI.
//

import Foundation
import Testing
@testable import VocalIsolation

@Suite("MDX on Core AI")
struct MDXSeparatorTests {

    private func fixture(_ name: String) throws -> [Float] {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "bin"))
        return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private var assetURL: URL {
        if let path = ProcessInfo.processInfo.environment["STEMS_MDX_ASSET"] { return URL(fileURLWithPath: path) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("stems").appendingPathComponent(MDXSeparator.assetName)
    }

    @Test("Chunks overlap by half and the last one reaches the end")
    func chunkLayout() {
        let C = VocalIsolator.chunk
        #expect(VocalIsolator.ChunkPlan(stereo: [[Float](repeating: 0, count: 1000), [Float](repeating: 0, count: 1000)]).starts == [0])
        let n = 2 * C + 1000
        let starts = VocalIsolator.ChunkPlan(stereo: [[Float](repeating: 0, count: n), [Float](repeating: 0, count: n)]).starts
        // 0, C/2, C by the stride; C + C falls short of n, so one more ends flush with it.
        #expect(starts == [0, C / 2, C, n - C])
    }

    @Test("separate() reproduces the Python reference vocal, closer than the 8-bit package, and repeats exactly")
    func endToEnd() async throws {
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            print("[skip] no MDX asset at \(assetURL.path)"); return
        }
        let n = 261_120
        let chunk = try fixture("e2e_chunk")
        let golden = try fixture("e2e_vocal")
        let channels = [Array(chunk[0..<n]), Array(chunk[n..<2 * n])]

        let separator = try await MDXSeparator(contentsOf: assetURL)
        let stems = try await separator.separate(channels, sampleRate: 44_100)
        var sig = 0.0, err = 0.0
        for ch in 0..<2 {
            for i in 10_000..<(n - 10_000) {
                let g = Double(golden[ch * n + i]), d = Double(stems.vocal[ch][i]) - g
                err += d * d; sig += g * g
            }
        }
        let snr = 10 * log10(sig / max(err, 1e-12))
        print("[mdx] Core AI vocal vs Python reference: \(String(format: "%.1f", snr)) dB")
        // The 8-bit Core ML package is held to 30 dB here; float16 measured 46.9.
        #expect(snr > 40, "vocal vs Python reference SNR = \(snr) dB")
        for i in 10_000..<(n - 10_000) {
            #expect(abs((stems.vocal[0][i] + stems.instrumental[0][i]) - channels[0][i]) < 1e-4)
        }
        let again = try await separator.separate(channels, sampleRate: 44_100)
        #expect(again.vocal == stems.vocal)
    }
}
