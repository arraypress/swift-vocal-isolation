//
//  MDXSeparator.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// Vocal / instrumental separation with MDX23C on Core AI.
///
/// The same network as ``VocalIsolator``'s Core ML package — UVR's
/// MDX23C-8KFFT-InstVoc_HQ — exported from the original checkpoint by
/// `Tools/export_mdx.py` in float16 instead of the 8-bit package, with the same
/// STFT, chunking and overlap-add: 2.3× faster on a 30 s mix (3.0 s against
/// 6.9 s) and the same vocal and instrumental scores to 0.01 dB. A separate type rather than an async overload on
/// ``VocalIsolator``: with both a sync and an async `separate`, Swift picked the
/// sync one even under `await`, and the Core ML path cannot run this asset.
///
/// ```swift
/// let separator = try await MDXSeparator(contentsOf: assetURL)   // stems-mdx23c-instvoc-float16.aimodel
/// let stems = try await separator.separate(contentsOf: songURL)
/// ```
public struct MDXSeparator {

    /// The asset's file name, as `stems` installs it.
    public static let assetName = "stems-mdx23c-instvoc-float16.aimodel"

    private let model: VocalModelCoreAI
    private let stft: VocalSTFT

    public init(contentsOf url: URL) async throws {
        guard let stft = VocalSTFT() else { throw VocalIsolationError.dspSetupFailed }
        self.stft = stft
        self.model = try await VocalModelCoreAI(contentsOf: url)
    }

    /// Separates de-interleaved channels at any rate; mono is duplicated.
    /// Stems come back at the input rate. A chunk the model fails on is an
    /// error, not the silence ``VocalIsolator`` leaves.
    public func separate(_ channels: [[Float]], sampleRate: Double, progress: ((Double) -> Void)? = nil) async throws -> VocalIsolator.Stems {
        let stereo = try VocalIsolator.stereoAtModelRate(channels, sampleRate: sampleRate)
        var plan = VocalIsolator.ChunkPlan(stereo: stereo)
        for (index, start) in plan.starts.enumerated() {
            try Task.checkCancellation()
            let (spec, frames, valid) = VocalIsolator.chunkSpectrogram(stereo, start: start, stft: stft)
            plan.add(try await model.predict(spec), frames: frames, start: start, valid: valid, stft: stft)
            progress?(Double(index + 1) / Double(plan.starts.count))
        }
        let (vocal, instrumental) = plan.finish(stereo)
        return VocalIsolator.fromModelRate(vocal: vocal, instrumental: instrumental, sampleRate: sampleRate, length: channels[0].count)
    }

    /// Reads an audio or video file and separates it.
    public func separate(contentsOf url: URL, progress: ((Double) -> Void)? = nil) async throws -> VocalIsolator.Stems {
        let (channels, rate) = try VocalIsolator.readAudio(url)
        return try await separate(channels, sampleRate: rate, progress: progress)
    }
}
