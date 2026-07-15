//
//  DeReverbTests.swift
//  VocalIsolationTests
//
//  End-to-end: the Swift de-reverb (STFT → Core ML net → ISTFT) must reproduce
//  the Python reference dry stem. Gated on a local `dereverb_net.mlpackage`
//  beside this file (the 214 MB model isn't committed) — skips when absent.
//

import Foundation
import Testing
@testable import VocalIsolation

@Suite("De-reverb")
struct DeReverbTests {

    private func fixture(_ name: String) throws -> [Float] {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "bin"))
        return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private var localModelURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("dereverb_net.mlpackage")
    }

    @Test("process() reproduces the Python reference dry vocal on a real chunk")
    func endToEnd() throws {
        guard FileManager.default.fileExists(atPath: localModelURL.path) else {
            print("[skip] no local model at \(localModelURL.lastPathComponent)"); return
        }
        let n = 261_120
        let chunk = try fixture("dereverb_chunk")
        let golden = try fixture("dereverb_dry")
        let channels = [Array(chunk[0..<n]), Array(chunk[n..<2 * n])]

        let dereverb = try DeReverb(modelURL: localModelURL)
        let out = try dereverb.process(channels, sampleRate: 44_100)

        #expect(out.dry.count == 2)
        #expect(out.dry[0].count == n)

        // Interior SNR vs golden (skip the Hann-tapered chunk edges).
        var sig = 0.0, err = 0.0
        for ch in 0..<2 {
            for i in 10_000..<(n - 10_000) {
                let g = golden[ch * n + i]
                let d = Double(out.dry[ch][i]) - Double(g)
                err += d * d; sig += Double(g) * Double(g)
            }
        }
        let snr = 10 * log10(sig / max(err, 1e-12))
        #expect(snr > 30, "de-reverb dry vs Python reference SNR = \(snr) dB")

        // Dry + reverb should reconstruct the input vocal.
        for i in 10_000..<(n - 10_000) {
            #expect(abs((out.dry[0][i] + out.reverb[0][i]) - channels[0][i]) < 1e-4)
        }
    }
}
