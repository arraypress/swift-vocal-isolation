//
//  PipelineTests.swift
//  VocalIsolationTests
//
//  Created by David Sherlock on 2026.
//
//  End-to-end: the Swift pipeline (STFT → Core ML net → ISTFT) must reproduce
//  the Python reference. Gated on a local `mdx_net.mlpackage` beside this file
//  (the 214 MB model isn't committed) — skips when absent (e.g. CI).
//

import Foundation
import Testing
@testable import VocalIsolation

@Suite("Pipeline")
struct PipelineTests {

    private func fixture(_ name: String) throws -> [Float] {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "bin"))
        return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private var localModelURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("mdx_net.mlpackage")
    }

    @Test("separate() reproduces the Python reference vocal on a real chunk")
    func endToEnd() throws {
        guard FileManager.default.fileExists(atPath: localModelURL.path) else {
            print("[skip] no local model at \(localModelURL.lastPathComponent)"); return
        }
        let n = 261_120
        let chunk = try fixture("e2e_chunk")
        let golden = try fixture("e2e_vocal")
        let channels = [Array(chunk[0..<n]), Array(chunk[n..<2 * n])]

        let isolator = try VocalIsolator(modelURL: localModelURL)
        let stems = try isolator.separate(channels, sampleRate: 44_100)

        #expect(stems.vocal.count == 2)
        #expect(stems.vocal[0].count == n)
        // Vocal should be isolated (well below the mix), and match the reference.
        let vocalRMS = rms(stems.vocal[0][...])
        #expect(vocalRMS > 0.01 && vocalRMS < rms(channels[0][...]))

        // Interior SNR vs golden (skip the Hann-tapered chunk edges).
        var sig = 0.0, err = 0.0
        for ch in 0..<2 {
            for i in 10_000..<(n - 10_000) {
                let g = golden[ch * n + i]
                let d = Double(stems.vocal[ch][i]) - Double(g)
                err += d * d; sig += Double(g) * Double(g)
            }
        }
        let snr = 10 * log10(sig / max(err, 1e-12))
        #expect(snr > 30, "end-to-end vocal vs Python reference SNR = \(snr) dB")

        // Vocal + instrumental should reconstruct the mix.
        for i in 10_000..<(n - 10_000) {
            #expect(abs((stems.vocal[0][i] + stems.instrumental[0][i]) - channels[0][i]) < 1e-4)
        }
    }

    private func rms(_ x: ArraySlice<Float>) -> Float {
        x.isEmpty ? 0 : (x.reduce(0) { $0 + $1 * $1 } / Float(x.count)).squareRoot()
    }
}
