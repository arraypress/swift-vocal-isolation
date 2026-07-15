//
//  VocalSTFTTests.swift
//  VocalIsolationTests
//
//  Created by David Sherlock on 2026.
//

import Foundation
import Testing
@testable import VocalIsolation

private func loadFloats(_ name: String) throws -> [Float] {
    let url = try #require(Bundle.module.url(forResource: name, withExtension: "bin"))
    return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func snr(_ a: [Float], _ golden: [Float]) -> Double {
    var sig = 0.0, err = 0.0
    for i in 0..<min(a.count, golden.count) {
        let d = Double(a[i]) - Double(golden[i]); err += d * d
        sig += Double(golden[i]) * Double(golden[i])
    }
    return 10 * log10(sig / max(err, 1e-12))
}

@Suite("STFT")
struct VocalSTFTTests {

    private let n = 50176

    /// The Swift STFT must reproduce the model's PyTorch `STFT` class so the
    /// Core ML net receives exactly the spectrogram it was trained on.
    @Test("Forward STFT matches the model's spectrogram")
    func stftMatchesGolden() throws {
        let audio = try loadFloats("stft_audio")
        let golden = try loadFloats("stft_spec")
        let channels = [Array(audio[0..<n]), Array(audio[n..<2 * n])]
        let stft = try #require(VocalSTFT())
        let (spec, frames) = stft.forward(channels)
        #expect(frames == 50)
        #expect(spec.count == golden.count)
        #expect(snr(spec, golden) > 45, "STFT vs golden SNR = \(snr(spec, golden)) dB")
    }

    /// The inverse must reproduce the model's reconstruction (windowed OLA +
    /// squared-window normalization + center trim).
    @Test("Inverse STFT matches the model's reconstruction")
    func istftMatchesGolden() throws {
        let golden = try loadFloats("stft_spec")
        let goldenRecon = try loadFloats("stft_recon")
        let stft = try #require(VocalSTFT())
        let recon = stft.inverse(golden, frames: 50)
        let flat = recon[0] + recon[1]
        #expect(snr(flat, goldenRecon) > 45, "ISTFT vs golden SNR = \(snr(flat, goldenRecon)) dB")
    }

    @Test("STFT → ISTFT round-trips a signal")
    func roundTrip() throws {
        var gen = SystemRandomNumberGenerator()
        let sig = (0..<n).map { _ in Float.random(in: -0.1...0.1, using: &gen) }
        let stft = try #require(VocalSTFT())
        let (spec, frames) = stft.forward([sig, sig])
        let recon = stft.inverse(spec, frames: frames)
        // Interior only — the model's dim_f truncation caps fidelity ~46 dB.
        let interior = Array(recon[0][8192..<(n - 8192)])
        let ref = Array(sig[8192..<(n - 8192)])
        #expect(snr(interior, ref) > 30)
    }
}
