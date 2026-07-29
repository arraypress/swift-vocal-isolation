//
//  DeReverb.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

@preconcurrency import AVFoundation
import Foundation

/// On-device de-reverb for a vocal, via the MDX23C De-Reverb model.
///
/// Takes a (usually already-isolated) vocal and splits it into a **dry** stem
/// — the vocal with room/reverb/echo stripped — and the **reverb** tail that
/// was removed. Shares the exact STFT, model wrapper and chunked overlap-add
/// as ``VocalIsolator`` (same MDX23C architecture: `n_fft 8192`, `hop 1024`,
/// `chunk 261120`, spec `[1,4,4096,256]` → `[1,8,4096,256]`); only the weights
/// and stem labels differ (stem 0 = dry, stem 1 = reverb).
///
/// ```swift
/// let dereverb = try DeReverb(modelURL: modelURL)
/// let out = try dereverb.process(vocalChannels, sampleRate: 44_100)
/// // out.dry — the clean vocal;  out.reverb — the tail it removed
/// ```
public struct DeReverb {

    static let modelRate = 44_100.0
    static let chunk = 261_120        // 255 · 1024 → exactly 256 STFT frames
    static let overlap = 2            // 50 % chunk overlap for a smooth join

    private let model: VocalModel
    private let stft: VocalSTFT

    /// A de-reverbed pair, at the sample rate they were produced at.
    public struct DryReverb: Sendable {
        /// The vocal with reverb/echo removed.
        public let dry: [[Float]]
        /// The reverb/echo tail that was stripped out.
        public let reverb: [[Float]]
        public let sampleRate: Double
    }

    /// Loads the de-reverb model from a compiled `.mlmodelc` or an `.mlpackage`.
    public init(modelURL: URL) throws {
        guard let stft = VocalSTFT() else { throw VocalIsolationError.dspSetupFailed }
        self.stft = stft
        self.model = try VocalModel(modelURL: modelURL)
    }

    /// De-reverbs de-interleaved float channels. Mono is duplicated to stereo;
    /// any sample rate is resampled to 44.1 kHz for the pass and back.
    /// Returns stems at the **input** sample rate.
    public func process(_ channels: [[Float]], sampleRate: Double, progress: ((Double) -> Void)? = nil) throws -> DryReverb {
        guard let first = channels.first, !first.isEmpty else { throw VocalIsolationError.emptyAudio }

        var stereo = channels.count >= 2 ? [channels[0], channels[1]] : [channels[0], channels[0]]
        if sampleRate != Self.modelRate {
            stereo = [VocalIsolator.resample(stereo[0], from: sampleRate, to: Self.modelRate),
                      VocalIsolator.resample(stereo[1], from: sampleRate, to: Self.modelRate)]
        }

        let (dry, reverb) = try processAt44k(stereo, progress: progress)

        if sampleRate != Self.modelRate {
            let d = [VocalIsolator.resample(dry[0], from: Self.modelRate, to: sampleRate),
                     VocalIsolator.resample(dry[1], from: Self.modelRate, to: sampleRate)]
            let r = [VocalIsolator.resample(reverb[0], from: Self.modelRate, to: sampleRate),
                     VocalIsolator.resample(reverb[1], from: Self.modelRate, to: sampleRate)]
            return DryReverb(dry: matchLength(d, to: first.count), reverb: matchLength(r, to: first.count), sampleRate: sampleRate)
        }
        return DryReverb(dry: dry, reverb: reverb, sampleRate: sampleRate)
    }

    /// Reads an audio/video file, de-reverbs it, and returns the stems.
    public func process(contentsOf url: URL, progress: ((Double) -> Void)? = nil) throws -> DryReverb {
        let (channels, rate) = try VocalIsolator.readAudio(url)
        return try process(channels, sampleRate: rate, progress: progress)
    }

    // MARK: - Core (mirror of VocalIsolator.separateAt44k; stem 0 = dry)

    /// Throws `CancellationError` if the calling task is cancelled part-way — see the note on
    /// `VocalIsolator.separateAt44k`. De-reverb is a second full-length pass, so a cancel that
    /// only landed between the two stages would still leave the user waiting out the whole of
    /// this one.
    private func processAt44k(_ stereo: [[Float]], progress: ((Double) -> Void)?) throws -> (dry: [[Float]], reverb: [[Float]]) {
        let C = Self.chunk, hop = C / Self.overlap
        let n = stereo[0].count
        var window = [Float](repeating: 0, count: C)
        for i in 0..<C { window[i] = 0.5 - 0.5 * cos(2 * .pi * Float(i) / Float(C)) }

        var dry = [[Float]](repeating: [Float](repeating: 0, count: n), count: 2)
        var wsum = [Float](repeating: 0, count: n)

        var starts = Array(stride(from: 0, through: max(0, n - C), by: hop))
        if let last = starts.last, last + C < n { starts.append(n - C) }
        if starts.isEmpty { starts = [0] }

        for (index, s) in starts.enumerated() {
            try Task.checkCancellation()
            let valid = min(C, n - s)
            let left = paddedChunk(stereo[0], start: s, length: C, valid: valid)
            let right = paddedChunk(stereo[1], start: s, length: C, valid: valid)
            let (spec, frames) = stft.forward([left, right])
            guard let stems = try? model.predict(spec) else {
                // A dead chunk = silence in the output; still advance the
                // progress bar so a rare failure doesn't look like a hang.
                progress?(Double(index + 1) / Double(starts.count))
                continue
            }

            let stemSize = VocalModel.specChannels * VocalModel.dimF * VocalModel.frames
            let drySpec = Array(stems[0..<stemSize])            // stem 0 = dry
            let dchunk = stft.inverse(drySpec, frames: frames)   // [2, C]
            for i in 0..<valid {
                let w = window[i]
                dry[0][s + i] += dchunk[0][i] * w
                dry[1][s + i] += dchunk[1][i] * w
                wsum[s + i] += w
            }
            progress?(Double(index + 1) / Double(starts.count))
        }

        var reverb = [[Float]](repeating: [Float](repeating: 0, count: n), count: 2)
        for ch in 0..<2 {
            for i in 0..<n {
                let e = wsum[i]
                let d = e > 1e-6 ? dry[ch][i] / e : 0
                dry[ch][i] = d
                reverb[ch][i] = stereo[ch][i] - d              // residual = the removed tail
            }
        }
        return (dry, reverb)
    }

    private func paddedChunk(_ x: [Float], start: Int, length: Int, valid: Int) -> [Float] {
        var out = [Float](repeating: 0, count: length)
        for i in 0..<valid { out[i] = x[start + i] }
        return out
    }

    private func matchLength(_ channels: [[Float]], to count: Int) -> [[Float]] {
        channels.map { ch in
            if ch.count == count { return ch }
            if ch.count > count { return Array(ch[0..<count]) }
            return ch + [Float](repeating: 0, count: count - ch.count)
        }
    }
}
