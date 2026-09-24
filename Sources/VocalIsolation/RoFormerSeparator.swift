//
//  RoFormerSeparator.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//
//  BS-RoFormer (Lu, Wang, Kong, Hung — ByteDance, ICASSP 2024) on Core AI,
//  with the MIT weights ZFTurbo trained: drums, bass, other and vocals. The
//  band-split transformer lives in the .aimodel at the model's chunk; the
//  spectrogram, the complex mask, the inverse and the chunked inference are
//  upstream's `demix` line for line — 11-second chunks at a 50% step,
//  reflect padding by a border, a linear fade window whose first and last
//  batches lose their fade-in and fade-out, sum over count. Batches are
//  replayed as upstream forms them because the fade rule is per batch.
//

import CoreAI
import Foundation

/// Four stems from a stereo mix, with BS-RoFormer.
public final class RoFormerSeparator: @unchecked Sendable {

    public static let assetName = "stems-bs_roformer-float32.aimodel"
    public static let sampleRate = 44_100.0
    public static let sources = ["drums", "bass", "other", "vocals"]
    /// `config.audio.chunk_size`.
    public static let chunk = 485_100
    /// `config.inference.num_overlap`: the step is chunk / overlap.
    public static let overlap = 2
    public static let step = chunk / overlap
    public static let fade = chunk / 10
    public static let border = chunk - step
    /// `config.inference.batch_size`: how many chunks share one fade decision.
    public static let batchSize = 2
    static let frames = RoFormerSTFT.frames(forSamples: chunk)   // 1101

    public struct Stems: Sendable {
        public let stems: [[[Float]]]
        public let sampleRate: Double
        public subscript(_ name: String) -> [[Float]]? { RoFormerSeparator.sources.firstIndex(of: name).map { stems[$0] } }
    }

    public let url: URL
    public let verifying: Bool
    public private(set) var recomputedChunks = 0
    private let function: InferenceFunction
    private let stft: RoFormerSTFT

    public init(contentsOf url: URL, verifying: Bool = true) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw VocalIsolationError.modelNotFound(url.path) }
        guard let stft = RoFormerSTFT() else { throw VocalIsolationError.dspSetupFailed }
        self.url = url
        self.stft = stft
        self.verifying = verifying
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: url, options: options) } catch {
            throw VocalIsolationError.modelUnavailable("\(url.lastPathComponent): \(error)")
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw VocalIsolationError.modelUnavailable("no main function in \(url.lastPathComponent)")
        }
        self.function = function
    }

    // MARK: - Whole recordings

    public func separate(_ channels: [[Float]], sampleRate: Double, progress: ((Double) -> Void)? = nil) async throws -> Stems {
        guard let first = channels.first, !first.isEmpty else { throw VocalIsolationError.emptyAudio }
        var stereo = channels.count >= 2 ? [channels[0], channels[1]] : [channels[0], channels[0]]
        if sampleRate != Self.sampleRate { stereo = stereo.map { VocalIsolator.resample($0, from: sampleRate, to: Self.sampleRate) } }
        var stems = try await separateAt44k(stereo, progress: progress)
        if sampleRate != Self.sampleRate {
            stems = stems.map { stem in stem.map { ch in
                let back = VocalIsolator.resample(ch, from: Self.sampleRate, to: sampleRate)
                return Array(back.prefix(first.count)) + [Float](repeating: 0, count: max(0, first.count - back.count))
            } }
        }
        return Stems(stems: stems, sampleRate: sampleRate)
    }

    public func separate(contentsOf url: URL, progress: ((Double) -> Void)? = nil) async throws -> Stems {
        let (channels, rate) = try VocalIsolator.readAudio(url)
        return try await separate(channels, sampleRate: rate, progress: progress)
    }

    /// Upstream's `demix` in generic mode at 44.1 kHz: `[stem][channel][sample]`.
    public func separateAt44k(_ stereo: [[Float]], progress: ((Double) -> Void)? = nil) async throws -> [[[Float]]] {
        let originalLength = stereo[0].count
        let padded = originalLength > 2 * Self.border && Self.border > 0
        let mix = padded ? stereo.map { DemucsSTFT.reflectPadded($0, left: Self.border, right: Self.border) } : stereo
        let length = mix[0].count
        let S = Self.sources.count
        var result = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: length), count: 2), count: S)
        var counter = [Float](repeating: 0, count: length)
        let base = Self.fadeWindow
        var batch: [(start: Int, length: Int, input: [[Float]])] = []
        var i = 0
        let total = (length + Self.step - 1) / Self.step
        var done = 0
        while i < length {
            let chunkLength = min(Self.chunk, length - i)
            let part = mix.map { ch -> [Float] in
                let slice = Array(ch[i..<(i + chunkLength)])
                if chunkLength == Self.chunk { return slice }
                // Longer than half a chunk: reflect-padded on the right; else zeros.
                return chunkLength > Self.chunk / 2 ? DemucsSTFT.reflectPadded(slice, left: 0, right: Self.chunk - chunkLength)
                                                   : slice + [Float](repeating: 0, count: Self.chunk - chunkLength)
            }
            batch.append((i, chunkLength, part))
            i += Self.step
            if batch.count >= Self.batchSize || i >= length {
                var window = base
                if i - Self.step == 0 { for n in 0..<Self.fade { window[n] = 1 } }               // first chunk: no fade-in
                else if i >= length { for n in (Self.chunk - Self.fade)..<Self.chunk { window[n] = 1 } }   // last batch: no fade-out
                for item in batch {
                    try Task.checkCancellation()
                    let stems = try await separateChunk(item.input)
                    for s in 0..<S {
                        for ch in 0..<2 {
                            for n in 0..<item.length { result[s][ch][item.start + n] += stems[s][ch][n] * window[n] }
                        }
                    }
                    for n in 0..<item.length { counter[item.start + n] += window[n] }
                    done += 1
                    progress?(Double(done) / Double(total))
                }
                batch.removeAll()
            }
        }
        for s in 0..<S {
            for ch in 0..<2 {
                for n in 0..<length {
                    let v = result[s][ch][n] / counter[n]
                    result[s][ch][n] = v.isNaN ? 0 : v          // upstream's nan_to_num
                }
            }
        }
        if padded {
            return result.map { $0.map { Array($0[Self.border..<(Self.border + originalLength)]) } }
        }
        return result
    }

    // MARK: - One chunk

    /// The model on exactly one chunk: `[stem][channel][sample]`, verified by a second run when ``verifying``.
    public func separateChunk(_ mix: [[Float]]) async throws -> [[[Float]]] {
        precondition(mix.count == 2 && mix[0].count == Self.chunk && mix[1].count == Self.chunk)
        let (spec, frames) = stft.spectrogram(mix)
        var mask = try await network(spec, frames: frames)
        if verifying {
            var runs = [mask]
            var agreed = false
            for attempt in 1..<DemucsSeparator.verificationAttempts {
                let next = try await network(spec, frames: frames)
                if let match = runs.first(where: { $0 == next }) {
                    if attempt > 1 { recomputedChunks += 1 }
                    mask = match; agreed = true; break
                }
                runs.append(next)
            }
            if !agreed {
                let spread = runs.dropFirst().map { String(format: "%.0f dB", DemucsSeparator.psnr(runs[0], $0)) }.joined(separator: ", ")
                throw VocalIsolationError.modelUnavailable("\(DemucsSeparator.verificationAttempts) runs of one chunk never agreed (against the first: \(spread)); the GPU is not computing reliably — is another GPU job running?")
            }
        }
        let width = RoFormerSTFT.width
        return (0..<Self.sources.count).map { s in
            stft.inverse(spectrogram: spec, mask: Array(mask[(s * frames * width)..<((s + 1) * frames * width)]), frames: frames, length: Self.chunk)
        }
    }

    /// The mask the network returns for a spectrogram: `[stem][frames][width]` flat.
    private func network(_ spec: [Float], frames: Int) async throws -> [Float] {
        var mask = NDArray(shape: [1, Self.sources.count, frames, RoFormerSTFT.width], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(mask.mutableRawView(), for: "mask")
        do {
            _ = try await function.run(inputs: ["x": DemucsSeparator.array(spec, shape: [1, frames, RoFormerSTFT.width])],
                                       states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch {
            throw VocalIsolationError.modelUnavailable("inference failed: \(error)")
        }
        return DemucsSeparator.floats(mask)
    }

    /// `_getWindowingArray(chunk, fade)`: linspace fade-in and fade-out, ones between.
    static let fadeWindow: [Float] = {
        var w = [Float](repeating: 1, count: chunk)
        for n in 0..<fade {
            let ramp = Float(n) / Float(fade - 1)
            w[n] = ramp
            w[chunk - fade + n] = 1 - ramp
        }
        return w
    }()
}
