//
//  VocalModel.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

import CoreML
import Foundation

/// Wraps the MDX-Net Core ML core: one fixed-shape spectrogram in, the stacked
/// stem spectrograms out. Input `spec` `[1, 4, 4096, 256]`, output
/// `[1, 8, 4096, 256]` = 2 stems × `[4, 4096, 256]` (vocal, then instrumental).
final class VocalModel {

    static let frames = 256          // fixed by the converted model (chunk 261120)
    static let dimF = 4096
    static let specChannels = 4      // Lᵣ, Lᵢ, Rᵣ, Rᵢ
    static let outChannels = 8       // 2 stems × 4

    private let model: MLModel

    /// Loads a compiled `.mlmodelc`, or compiles an `.mlpackage`/`.mlmodel` on
    /// the fly and loads that.
    init(modelURL: URL) throws {
        let url = try Self.compiledURL(for: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Compiles once and caches in Caches/ keyed by name + mtime — compiling
    /// the .mlpackage on EVERY Music Mode run cost seconds each time.
    static func compiledURL(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" { return modelURL }
        let fm = FileManager.default
        let mtime = (try? fm.attributesOfItem(atPath: modelURL.path)[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let dir = try fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("swift-vocal-isolation/CompiledModels", isDirectory: true)
        let dst = dir.appendingPathComponent("\(modelURL.deletingPathExtension().lastPathComponent)-\(Int(mtime)).mlmodelc")
        if fm.fileExists(atPath: dst.path) { return dst }
        let compiled = try MLModel.compileModel(at: modelURL)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try? fm.removeItem(at: dst)
        try fm.moveItem(at: compiled, to: dst)
        return dst
    }

    /// `spec` is `[4, 4096, 256]` (row-major); returns `[8, 4096, 256]`.
    func predict(_ spec: [Float]) throws -> [Float] {
        let inCount = Self.specChannels * Self.dimF * Self.frames
        precondition(spec.count == inCount)
        let input = try MLMultiArray(shape: [1, NSNumber(value: Self.specChannels), NSNumber(value: Self.dimF), NSNumber(value: Self.frames)], dataType: .float32)
        let ip = input.dataPointer.assumingMemoryBound(to: Float.self)
        spec.withUnsafeBufferPointer { ip.update(from: $0.baseAddress!, count: inCount) }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["spec": input])
        let result = try model.prediction(from: provider)
        guard let name = result.featureNames.first,
              let out = result.featureValue(for: name)?.multiArrayValue else {
            throw VocalIsolationError.predictionFailed
        }
        let outCount = Self.outChannels * Self.dimF * Self.frames
        var stems = [Float](repeating: 0, count: outCount)
        if out.dataType == .float16 {
            let op = out.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<outCount { stems[i] = Float(op[i]) }
        } else {
            let op = out.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<outCount { stems[i] = op[i] }
        }
        return stems
    }
}
