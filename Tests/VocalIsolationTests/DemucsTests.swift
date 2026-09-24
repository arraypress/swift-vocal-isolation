//
//  DemucsTests.swift
//  VocalIsolationTests
//
//  Created by David Sherlock on 2026.
//
//  The Demucs port against upstream's own numbers: the STFT pair on fixtures
//  in the repo, then — when STEMS_DEMUCS_REFS points at a reference folder
//  written by Tools/export_demucs.py and the asset is installed — one
//  training segment through the model, and whole clips end to end.
//

import Foundation
import Testing
@testable import VocalIsolation

private func demucsFixture(_ name: String, _ ext: String) throws -> [Float] {
    let url = try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Demucs"))
    return try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func floats(at path: String) throws -> [Float] {
    try Data(contentsOf: URL(fileURLWithPath: path)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// Peak signal-to-noise ratio of `got` against `reference`, in dB.
private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    precondition(reference.count == got.count)
    var err = 0.0
    var lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices {
        let d = Double(reference[i]) - Double(got[i]); err += d * d
        lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i]))
    }
    let rms = (err / Double(reference.count)).squareRoot()
    return rms == 0 ? .infinity : 20 * log10((hi - lo) / rms)
}

private func referenceFolder() -> String? { ProcessInfo.processInfo.environment["STEMS_DEMUCS_REFS"] }

private func installedAsset() -> URL? {
    let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("stems").appendingPathComponent(DemucsSeparator.assetName)
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}

struct DemucsTests {

    // MARK: - STFT, no model

    @Test func spectrogramMatchesUpstream() throws {
        let input = try demucsFixture("short_in", "f32")           // [2][22050]
        let reference = try demucsFixture("short_mag", "f32")      // [4][2048][22]
        let stft = try #require(DemucsSTFT())
        let n = input.count / 2
        let (ours, frames) = stft.spectrogram([Array(input[0..<n]), Array(input[n..<(2 * n)])])
        #expect(frames == 22)
        #expect(ours.count == reference.count)
        let db = psnr(reference, ours)
        print("DEMUCS spectrogram PSNR \(db) dB")
        #expect(db > 90)
    }

    @Test func inverseMatchesUpstream() throws {
        let cac = try demucsFixture("cac_in", "f32")               // [4][2048][22], one stem
        let reference = try demucsFixture("cac_out", "f32")        // [2][22050]
        let stft = try #require(DemucsSTFT())
        let out = stft.inverse(cac, frames: 22, length: 22050)
        let db = psnr(reference, out[0] + out[1])
        print("DEMUCS inverse PSNR \(db) dB")
        #expect(db > 90)
    }

    @Test func transitionWeightIsUpstreams() {
        let w = DemucsSeparator.transitionWeight
        #expect(w.count == DemucsSeparator.segment)
        #expect(w.first == 1 / Float(DemucsSeparator.segment / 2))
        #expect(w.max() == 1)
        #expect(w[DemucsSeparator.segment / 2 - 1] == 1)
        #expect(DemucsSeparator.chunkStride == 257_985)
    }

    // MARK: - Model

    @Test func segmentMatchesUpstream() async throws {
        guard let refs = referenceFolder(), let asset = installedAsset() else { return }
        let separator = try await DemucsSeparator(contentsOf: asset)
        let L = DemucsSeparator.segment
        for clip in ["demo", "kit"] where FileManager.default.fileExists(atPath: "\(refs)/\(clip)/chunk_mix.f32") {
            let mix = try floats(at: "\(refs)/\(clip)/chunk_mix.f32")
            let stems = try await separator.separateSegment([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
            let reference = try floats(at: "\(refs)/\(clip)/chunk_out.f32")   // [4][2][L]
            let db = psnr(reference, stems.flatMap { $0.flatMap { $0 } })
            print("DEMUCS segment (\(clip)) PSNR \(db) dB")
            #expect(db > 60)
        }
        // Every chunk of a clip, when the reference folder has them: chunks/<offset>_mix.f32 and _out.f32.
        for clip in ["kit"] {
            let dir = "\(refs)/\(clip)/chunks"
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
            var worst = Double.infinity
            for name in names.filter({ $0.hasSuffix("_mix.f32") }).sorted() {
                let mix = try floats(at: "\(dir)/\(name)")
                let stems = try await separator.separateSegment([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
                let reference = try floats(at: "\(dir)/\(name.replacingOccurrences(of: "_mix", with: "_out"))")
                let flat = stems.flatMap { $0.flatMap { $0 } }
                var perStem: [String] = []
                for s in 0..<4 {
                    let db = psnr(Array(reference[(s * 2 * L)..<((s + 1) * 2 * L)]), Array(flat[(s * 2 * L)..<((s + 1) * 2 * L)]))
                    perStem.append(String(format: "%@ %.0f", DemucsSeparator.sources[s], db)); worst = min(worst, db)
                }
                print("DEMUCS chunk \(name.replacingOccurrences(of: "_mix.f32", with: "")): \(perStem.joined(separator: ", ")) dB")
            }
            print("DEMUCS worst chunk-stem PSNR on \(clip): \(worst) dB")
        }
    }

    @Test func clipsMatchUpstream() async throws {
        guard let refs = referenceFolder(), let asset = installedAsset() else { return }
        let separator = try await DemucsSeparator(contentsOf: asset)
        for clip in ["demo", "kit"] where FileManager.default.fileExists(atPath: "\(refs)/\(clip)/stems.f32") {
            let (channels, rate) = try VocalIsolator.readAudio(URL(fileURLWithPath: "\(refs)/\(clip)/audio_44k.wav"))
            #expect(rate == 44_100)
            let stems = try await separator.separateAt44k([channels[0], channels[1]])
            if ProcessInfo.processInfo.environment["STEMS_DEMUCS_DUMP"] != nil {
                let flat = stems.flatMap { $0.flatMap { $0 } }
                try flat.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: "\(refs)/\(clip)/ours_stems.f32"))
            }
            let reference = try floats(at: "\(refs)/\(clip)/stems.f32")      // [4][2][samples]
            let n = channels[0].count
            #expect(reference.count == 4 * 2 * n)
            for (s, name) in DemucsSeparator.sources.enumerated() {
                let ref = Array(reference[(s * 2 * n)..<((s + 1) * 2 * n)])
                let db = psnr(ref, stems[s][0] + stems[s][1])
                print("DEMUCS \(clip) \(name): PSNR \(db) dB over \(n) samples")
                #expect(db > 60, "\(clip) \(name)")
            }
            print("DEMUCS \(clip): verification recomputed \(separator.recomputedSegments) segment(s)")
        }
    }

    /// The same segment several times on each compute unit: how repeatable is the model, and how fast.
    @Test func segmentRepeatability() async throws {
        guard let refs = referenceFolder(), let asset = installedAsset() else { return }
        let L = DemucsSeparator.segment
        let mix = try floats(at: "\(refs)/kit/chunk_mix.f32")
        let input = [Array(mix[0..<L]), Array(mix[L..<(2 * L)])]
        let reference = try floats(at: "\(refs)/kit/chunk_out.f32")
        // GPU only: Core AI's CPU path segfaults on this graph (measured 2026-09-24), and a crash takes the whole run down.
        for unit in [DemucsSeparator.ComputeUnit.gpu] {
            let separator = try await DemucsSeparator(contentsOf: asset, computeUnit: unit)
            var runs: [[Float]] = []
            var seconds: [Double] = []
            for _ in 0..<3 {
                let started = Date()
                runs.append(try await separator.separateSegment(input).flatMap { $0.flatMap { $0 } })
                seconds.append(Date().timeIntervalSince(started))
            }
            let identical = runs.allSatisfy { $0 == runs[0] }
            let pair = psnr(runs[0], runs[1])
            let vsUpstream = runs.map { psnr(reference, $0) }
            print(String(format: "DEMUCS %@: runs identical %@, run-to-run PSNR %.0f dB, vs upstream %@ dB, %.2f s per segment",
                         unit.rawValue, identical ? "yes" : "no", pair, vsUpstream.map { String(format: "%.0f", $0) }.joined(separator: "/"), seconds.dropFirst().reduce(0, +) / Double(max(1, seconds.count - 1))))
        }
    }

    /// Diagnostics: is the run-to-run variation on long clips timing-related?
    @Test func chunkSequenceTiming() async throws {
        guard ProcessInfo.processInfo.environment["STEMS_DEMUCS_TIMING"] != nil, let refs = referenceFolder(), let asset = installedAsset() else { return }
        let L = DemucsSeparator.segment
        let dir = "\(refs)/kit/chunks"
        let names = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix("_mix.f32") }.sorted()
        var inputs: [[[Float]]] = [], outputs: [[Float]] = []
        for name in names {
            let mix = try floats(at: "\(dir)/\(name)"); inputs.append([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
            outputs.append(try floats(at: "\(dir)/\(name.replacingOccurrences(of: "_mix", with: "_out"))"))
        }
        let separator = try await DemucsSeparator(contentsOf: asset)
        for pause in [nil, Duration.milliseconds(300)] {
            var worst = Double.infinity, sum = 0.0
            for (i, input) in inputs.enumerated() {
                let got = try await separator.separateSegment(input).flatMap { $0.flatMap { $0 } }
                if let pause { try await Task.sleep(for: pause) }
                let db = psnr(outputs[i], got); worst = min(worst, db); sum += db
            }
            print(String(format: "DEMUCS sequence pause %@: worst %.0f dB, mean %.0f dB over %d chunks", pause == nil ? "none" : "300 ms", worst, sum / Double(inputs.count), inputs.count))
        }
        // Whole clip with pauses between chunks.
        let (channels, _) = try VocalIsolator.readAudio(URL(fileURLWithPath: "\(refs)/kit/audio_44k.wav"))
        let reference = try floats(at: "\(refs)/kit/stems.f32"); let n = channels[0].count
        for pause in [Duration.milliseconds(300), nil] {
            DemucsSeparator.debugPause = pause
            let stems = try await separator.separateAt44k([channels[0], channels[1]])
            let dbs = (0..<4).map { s in psnr(Array(reference[(s * 2 * n)..<((s + 1) * 2 * n)]), stems[s][0] + stems[s][1]) }
            print(String(format: "DEMUCS whole clip pause %@: %@ dB", pause == nil ? "none" : "300 ms", dbs.map { String(format: "%.0f", $0) }.joined(separator: "/")))
        }
        DemucsSeparator.debugPause = nil
    }

    /// Diagnostics: which chunks go wrong, are they wrong in isolation, and is it state carried between runs?
    @Test func chunkFaultLocalisation() async throws {
        guard ProcessInfo.processInfo.environment["STEMS_DEMUCS_FAULT"] != nil, let refs = referenceFolder(), let asset = installedAsset() else { return }
        let L = DemucsSeparator.segment
        let dir = "\(refs)/kit/chunks"
        let names = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix("_mix.f32") }.sorted()
        var inputs: [[[Float]]] = [], outputs: [[Float]] = []
        for name in names {
            let mix = try floats(at: "\(dir)/\(name)"); inputs.append([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
            outputs.append(try floats(at: "\(dir)/\(name.replacingOccurrences(of: "_mix", with: "_out"))"))
        }
        func rms(_ x: [Float]) -> Float { (x.reduce(0) { $0 + $1 * $1 } / Float(x.count)).squareRoot() }
        let separator = try await DemucsSeparator(contentsOf: asset)
        var worst: [(Int, Double)] = []
        for pass in 0..<4 {
            DemucsSeparator.debugReadPause = pass >= 2 ? .milliseconds(200) : nil
            var line: [String] = []
            for (i, input) in inputs.enumerated() {
                let got = try await separator.separateSegment(input).flatMap { $0.flatMap { $0 } }
                let db = psnr(outputs[i], got)
                if db < 125 { line.append(String(format: "#%d %.0f dB (mix rms %.3f)", i, db, rms(input[0]))); worst.append((i, db)) }
            }
            print("DEMUCS pass \(pass) (read pause \(DemucsSeparator.debugReadPause == nil ? "none" : "200 ms")) chunks under 125 dB: \(line.isEmpty ? "none" : line.joined(separator: ", "))")
        }
        DemucsSeparator.debugReadPause = nil
        guard let target = worst.min(by: { $0.1 < $1.1 })?.0 else { return }
        // The worst chunk alone, repeatedly, on a fresh instance.
        let fresh = try await DemucsSeparator(contentsOf: asset)
        var alone: [String] = []
        for _ in 0..<4 { alone.append(String(format: "%.0f", psnr(outputs[target], try await fresh.separateSegment(inputs[target]).flatMap { $0.flatMap { $0 } }))) }
        print("DEMUCS chunk #\(target) alone ×4 on a fresh instance: \(alone.joined(separator: "/")) dB")
        // Alternating with a different chunk.
        let other = (target + 7) % inputs.count
        var alternating: [String] = []
        for _ in 0..<4 {
            _ = try await fresh.separateSegment(inputs[other])
            alternating.append(String(format: "%.0f", psnr(outputs[target], try await fresh.separateSegment(inputs[target]).flatMap { $0.flatMap { $0 } })))
        }
        print("DEMUCS chunk #\(target) after chunk #\(other), ×4: \(alternating.joined(separator: "/")) dB")
    }

    /// Diagnostics: where the time goes in one segment, and how often a chunk comes out wrong over many passes.
    @Test func faultRate() async throws {
        guard let passes = ProcessInfo.processInfo.environment["STEMS_DEMUCS_PASSES"].flatMap(Int.init), let refs = referenceFolder(), let asset = installedAsset() else { return }
        let L = DemucsSeparator.segment
        let stft = try #require(DemucsSTFT())
        let mix = try floats(at: "\(refs)/kit/chunk_mix.f32"); let input = [Array(mix[0..<L]), Array(mix[L..<(2 * L)])]
        var t = Date(); let (mag, frames) = stft.spectrogram(input); let stftSeconds = Date().timeIntervalSince(t)
        t = Date(); _ = stft.inverse(Array(mag[0..<(4 * DemucsSTFT.bins * frames)]), frames: frames, length: L); let istftSeconds = Date().timeIntervalSince(t)
        let separator = try await DemucsSeparator(contentsOf: asset)
        _ = try await separator.separateSegment(input)
        t = Date(); _ = try await separator.separateSegment(input); let segmentSeconds = Date().timeIntervalSince(t)
        print(String(format: "DEMUCS timing: stft %.2f s, one istft %.2f s (×4 per segment), whole segment %.2f s", stftSeconds, istftSeconds, segmentSeconds))
        let dir = "\(refs)/kit/chunks"
        let names = try FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix("_mix.f32") }.sorted()
        var inputs: [[[Float]]] = [], outputs: [[Float]] = []
        for name in names {
            let m = try floats(at: "\(dir)/\(name)"); inputs.append([Array(m[0..<L]), Array(m[L..<(2 * L)])])
            outputs.append(try floats(at: "\(dir)/\(name.replacingOccurrences(of: "_mix", with: "_out"))"))
        }
        var faults: [String] = []
        for pass in 0..<passes {
            for (i, input) in inputs.enumerated() {
                let db = psnr(outputs[i], try await separator.separateSegment(input).flatMap { $0.flatMap { $0 } })
                if db < 125 { faults.append(String(format: "pass %d #%d %.0f dB", pass, i, db)) }
            }
        }
        print("DEMUCS faults over \(passes) passes × \(inputs.count) chunks: \(faults.count) — \(faults.joined(separator: ", "))")
    }

    // MARK: - The fine-tuned bag

    private func installedBag() -> URL? {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("stems").appendingPathComponent(DemucsSeparator.assetName(forVariant: "htdemucs_ft"))
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// htdemucs_ft: four models, stem k from model k, against references from `export_demucs.py --name htdemucs_ft`.
    @Test func bagMatchesUpstream() async throws {
        guard let refs = ProcessInfo.processInfo.environment["STEMS_DEMUCS_REFS_FT"], let asset = installedBag() else { return }
        let separator = try await DemucsSeparator(contentsOf: asset)
        #expect(separator.isBag)
        let L = DemucsSeparator.segment
        for clip in ["demo", "kit"] where FileManager.default.fileExists(atPath: "\(refs)/\(clip)/chunk_mix.f32") {
            let mix = try floats(at: "\(refs)/\(clip)/chunk_mix.f32")
            let stems = try await separator.separateSegment([Array(mix[0..<L]), Array(mix[L..<(2 * L)])])
            let reference = try floats(at: "\(refs)/\(clip)/chunk_out.f32")
            let db = psnr(reference, stems.flatMap { $0.flatMap { $0 } })
            print("DEMUCS ft segment (\(clip)) PSNR \(db) dB")
            #expect(db > 60)
        }
        for clip in ["demo", "kit"] where FileManager.default.fileExists(atPath: "\(refs)/\(clip)/stems.f32") {
            let (channels, _) = try VocalIsolator.readAudio(URL(fileURLWithPath: "\(refs)/\(clip)/audio_44k.wav"))
            let stems = try await separator.separateAt44k([channels[0], channels[1]])
            let reference = try floats(at: "\(refs)/\(clip)/stems.f32")
            let n = channels[0].count
            for (s, name) in DemucsSeparator.sources.enumerated() {
                let db = psnr(Array(reference[(s * 2 * n)..<((s + 1) * 2 * n)]), stems[s][0] + stems[s][1])
                print("DEMUCS ft \(clip) \(name): PSNR \(db) dB")
                #expect(db > 60, "\(clip) \(name)")
            }
            print("DEMUCS ft \(clip): verification recomputed \(separator.recomputedSegments) model run(s)")
        }
    }
}
