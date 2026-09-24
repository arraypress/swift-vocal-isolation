//
//  RoFormerTests.swift
//  VocalIsolationTests
//
//  Created by David Sherlock on 2026.
//
//  The BS-RoFormer port against upstream: the spectrogram layout and the
//  masked inverse on fixtures in the repo, then — with STEMS_ROFORMER_REFS
//  pointing at a folder written by the reference harness and the asset
//  installed — every chunk through the model and whole clips end to end.
//

import Foundation
import Testing
@testable import VocalIsolation

private func fixture(_ name: String, _ ext: String) throws -> [Float] {
    let url = try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "RoFormer"))
    return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func floats(at path: String) throws -> [Float] {
    try Data(contentsOf: URL(fileURLWithPath: path)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    precondition(reference.count == got.count)
    var err = 0.0, lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices {
        let d = Double(reference[i]) - Double(got[i]); err += d * d
        lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i]))
    }
    let rms = (err / Double(reference.count)).squareRoot()
    return rms == 0 ? .infinity : 20 * log10((hi - lo) / rms)
}

private func referenceFolder() -> String? { ProcessInfo.processInfo.environment["STEMS_ROFORMER_REFS"] }

private func installedAsset() -> URL? {
    let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("stems").appendingPathComponent(RoFormerSeparator.assetName)
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}

struct RoFormerTests {

    // MARK: - No model needed

    @Test func spectrogramLayoutMatchesUpstream() throws {
        let input = try fixture("short_in", "f32")                 // [2][44100]
        let reference = try fixture("short_x", "f32")              // [101][4100]
        let stft = try #require(RoFormerSTFT())
        let n = input.count / 2
        let (x, frames) = stft.spectrogram([Array(input[0..<n]), Array(input[n..<(2 * n)])])
        #expect(frames == 101)
        #expect(x.count == reference.count)
        let db = psnr(reference, x)
        print("ROFORMER spectrogram PSNR \(db) dB")
        #expect(db > 90)
    }

    @Test func maskedInverseMatchesUpstream() throws {
        let input = try fixture("short_in", "f32")
        let mask = try fixture("mask", "f32")                      // [101][4100], one stem
        let reference = try fixture("masked_out", "f32")           // [2][44100]
        let stft = try #require(RoFormerSTFT())
        let n = input.count / 2
        let (x, frames) = stft.spectrogram([Array(input[0..<n]), Array(input[n..<(2 * n)])])
        let out = stft.inverse(spectrogram: x, mask: mask, frames: frames, length: n)
        let db = psnr(reference, out[0] + out[1])
        print("ROFORMER masked inverse PSNR \(db) dB")
        #expect(db > 90)
    }

    @Test func fadeWindowIsUpstreams() {
        let w = RoFormerSeparator.fadeWindow
        #expect(w.count == RoFormerSeparator.chunk)
        #expect(w[0] == 0); #expect(w[RoFormerSeparator.fade - 1] == 1); #expect(w[RoFormerSeparator.chunk / 2] == 1)
        #expect(w[RoFormerSeparator.chunk - RoFormerSeparator.fade] == 1); #expect(w[RoFormerSeparator.chunk - 1] == 0)
        #expect(RoFormerSeparator.step == 242_550 && RoFormerSeparator.border == 242_550 && RoFormerSeparator.fade == 48_510)
    }

    // MARK: - Model

    @Test func chunksMatchUpstream() async throws {
        guard let refs = referenceFolder(), let asset = installedAsset() else { return }
        let separator = try await RoFormerSeparator(contentsOf: asset)
        let L = RoFormerSeparator.chunk
        for clip in ["demo", "kit"] {
            let dir = "\(refs)/\(clip)/chunks"
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
            var worst = Double.infinity
            for name in names.filter({ $0.hasSuffix("_in.f32") }).sorted() {
                let mix = try floats(at: "\(dir)/\(name)")
                let stems = try await separator.separateChunk([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
                let reference = try floats(at: "\(dir)/\(name.replacingOccurrences(of: "_in", with: "_out"))")   // [4][2][L]
                let flat = stems.flatMap { $0.flatMap { $0 } }
                var per: [String] = []
                for s in 0..<4 {
                    let db = psnr(Array(reference[(s * 2 * L)..<((s + 1) * 2 * L)]), Array(flat[(s * 2 * L)..<((s + 1) * 2 * L)]))
                    per.append(String(format: "%@ %.0f", RoFormerSeparator.sources[s], db)); worst = min(worst, db)
                }
                print("ROFORMER \(clip) chunk \(name.replacingOccurrences(of: "_in.f32", with: "")): \(per.joined(separator: ", ")) dB")
            }
            print("ROFORMER \(clip): worst chunk-stem PSNR \(worst) dB, verification recomputed \(separator.recomputedChunks)")
            #expect(worst > 60)
        }
    }

    @Test func clipsMatchUpstream() async throws {
        guard let refs = referenceFolder(), let asset = installedAsset() else { return }
        let separator = try await RoFormerSeparator(contentsOf: asset)
        for clip in ["demo", "kit"] where FileManager.default.fileExists(atPath: "\(refs)/\(clip)/stems.f32") {
            let (channels, rate) = try VocalIsolator.readAudio(URL(fileURLWithPath: "\(refs)/\(clip)/audio_44k.wav"))
            #expect(rate == 44_100)
            let stems = try await separator.separateAt44k([channels[0], channels[1]])
            let reference = try floats(at: "\(refs)/\(clip)/stems.f32")
            let n = channels[0].count
            #expect(reference.count == 4 * 2 * n)
            for (s, name) in RoFormerSeparator.sources.enumerated() {
                let db = psnr(Array(reference[(s * 2 * n)..<((s + 1) * 2 * n)]), stems[s][0] + stems[s][1])
                print("ROFORMER \(clip) \(name): PSNR \(db) dB over \(n) samples")
                #expect(db > 60, "\(clip) \(name)")
            }
            print("ROFORMER \(clip): verification recomputed \(separator.recomputedChunks) chunk(s)")
        }
    }
}
