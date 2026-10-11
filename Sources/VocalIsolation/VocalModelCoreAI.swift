//
//  VocalModelCoreAI.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

import CoreAI
import Foundation

/// The MDX23C core as a Core AI asset: the same cut as ``VocalModel`` — `spec`
/// `[1, 4, 4096, 256]` in, `stems` `[1, 8, 4096, 256]` out — so the STFT and the
/// chunking are shared.
///
/// Exported from UVR's MDX23C-8KFFT-InstVoc_HQ checkpoint by `Tools/export_mdx.py`.
/// Measured per chunk against torch's float32 on this machine: the float16 asset
/// (214 MB, the one shipped) 62.6 dB in 0.27 s, the float32 one (427 MB) 123.9 dB
/// in 0.31 s, the 8-bit Core ML package it replaces 35.3 dB in 0.74 s. Repeated
/// runs are bit-for-bit identical — none of the wrong segments Demucs shows — and
/// on three 30 s mixes with a known vocal all three scored the same to 0.01 dB.
struct VocalModelCoreAI: Sendable {

    private let function: InferenceFunction

    init(contentsOf url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw VocalIsolationError.modelNotFound(url.path) }
        let model: AIModel
        do { model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu)) } catch {
            throw VocalIsolationError.modelUnavailable("\(url.lastPathComponent): \(error)")
        }
        guard let function = try model.loadFunction(named: "main") else {
            throw VocalIsolationError.modelUnavailable("no main function in \(url.lastPathComponent)")
        }
        self.function = function
    }

    /// `spec` is `[4, 4096, 256]` (row-major); returns `[8, 4096, 256]`.
    func predict(_ spec: [Float]) async throws -> [Float] {
        var stems = NDArray(shape: [1, VocalModel.outChannels, VocalModel.dimF, VocalModel.frames], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(stems.mutableRawView(), for: "stems")
        do {
            _ = try await function.run(
                inputs: ["spec": DemucsSeparator.array(spec, shape: [1, VocalModel.specChannels, VocalModel.dimF, VocalModel.frames])],
                states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch {
            throw VocalIsolationError.modelUnavailable("inference failed: \(error)")
        }
        return DemucsSeparator.floats(stems)
    }
}
