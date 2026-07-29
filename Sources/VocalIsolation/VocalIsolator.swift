//
//  VocalIsolator.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// Errors from vocal isolation.
public enum VocalIsolationError: Error {
    case modelLoadFailed
    case dspSetupFailed
    case predictionFailed
    case audioReadFailed
    case emptyAudio
}

/// On-device vocal / instrumental separation with MDX-Net via Core ML.
///
/// Splits a song into a clean **vocal** stem and its **instrumental** — feed the
/// vocal to a speech recognizer for lyric captions, or keep the stems as an
/// acapella / producer tool. Everything runs on-device; the model is 44.1 kHz
/// stereo, and audio at other rates/layouts is converted in and out.
///
/// ```swift
/// let isolator = try VocalIsolator(modelURL: modelURL)
/// let stems = try isolator.separate(channels, sampleRate: 44_100)
/// // stems.vocal, stems.instrumental — each [[Float]] (L, R)
/// ```
public struct VocalIsolator {

    /// The model's native rate and chunking.
    static let modelRate = 44_100.0
    static let chunk = 261_120        // 255 · 1024 → exactly 256 STFT frames
    static let overlap = 2            // 50 % chunk overlap for a smooth join

    private let model: VocalModel
    private let stft: VocalSTFT

    /// A separated pair, at the sample rate they were produced at.
    public struct Stems: Sendable {
        public let vocal: [[Float]]
        public let instrumental: [[Float]]
        public let sampleRate: Double
    }

    /// Loads the model from a compiled `.mlmodelc` or an `.mlpackage`/`.mlmodel`.
    public init(modelURL: URL) throws {
        guard let stft = VocalSTFT() else { throw VocalIsolationError.dspSetupFailed }
        self.stft = stft
        self.model = try VocalModel(modelURL: modelURL)
    }

    // MARK: - Core

    /// Separates de-interleaved float channels. Mono is duplicated to stereo;
    /// any sample rate is resampled to 44.1 kHz for the pass and back.
    /// Returns stems at the **input** sample rate.
    public func separate(_ channels: [[Float]], sampleRate: Double, progress: ((Double) -> Void)? = nil) throws -> Stems {
        guard let first = channels.first, !first.isEmpty else { throw VocalIsolationError.emptyAudio }

        // Stereo at 44.1 kHz.
        var stereo = channels.count >= 2 ? [channels[0], channels[1]] : [channels[0], channels[0]]
        if sampleRate != Self.modelRate {
            stereo = [Self.resample(stereo[0], from: sampleRate, to: Self.modelRate),
                      Self.resample(stereo[1], from: sampleRate, to: Self.modelRate)]
        }

        let (vocal, instrumental) = try separateAt44k(stereo, progress: progress)

        // Back to the input rate.
        if sampleRate != Self.modelRate {
            let v = [Self.resample(vocal[0], from: Self.modelRate, to: sampleRate), Self.resample(vocal[1], from: Self.modelRate, to: sampleRate)]
            let inst = [Self.resample(instrumental[0], from: Self.modelRate, to: sampleRate), Self.resample(instrumental[1], from: Self.modelRate, to: sampleRate)]
            return Stems(vocal: matchLength(v, to: first.count), instrumental: matchLength(inst, to: first.count), sampleRate: sampleRate)
        }
        return Stems(vocal: vocal, instrumental: instrumental, sampleRate: sampleRate)
    }

    /// Throws `CancellationError` if the calling task is cancelled part-way.
    ///
    /// The check has to live *here*, per chunk. This is the only loop in the pass, so a caller
    /// that wraps `separate` in a `Task` and cancels it has no other point of leverage: without
    /// this, cancelling a five-minute track keeps every core busy until the whole separation
    /// finishes, and the caller's own `Task.checkCancellation()` only runs once it is already
    /// too late to matter.
    ///
    /// Outside a task context `Task.isCancelled` is simply `false`, so synchronous callers are
    /// unaffected.
    private func separateAt44k(_ stereo: [[Float]], progress: ((Double) -> Void)?) throws -> (vocal: [[Float]], instrumental: [[Float]]) {
        let C = Self.chunk, hop = C / Self.overlap
        let n = stereo[0].count
        var window = [Float](repeating: 0, count: C)
        for i in 0..<C { window[i] = 0.5 - 0.5 * cos(2 * .pi * Float(i) / Float(C)) }

        var vocal = [[Float]](repeating: [Float](repeating: 0, count: n), count: 2)
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
            let vocalSpec = Array(stems[0..<stemSize])          // stem 0 = vocal
            let vchunk = stft.inverse(vocalSpec, frames: frames) // [2, C]
            for i in 0..<valid {
                let w = window[i]
                vocal[0][s + i] += vchunk[0][i] * w
                vocal[1][s + i] += vchunk[1][i] * w
                wsum[s + i] += w
            }
            progress?(Double(index + 1) / Double(starts.count))
        }

        var instrumental = [[Float]](repeating: [Float](repeating: 0, count: n), count: 2)
        for ch in 0..<2 {
            for i in 0..<n {
                let e = wsum[i]
                let v = e > 1e-6 ? vocal[ch][i] / e : 0
                vocal[ch][i] = v
                instrumental[ch][i] = stereo[ch][i] - v    // residual = everything but the vocal
            }
        }
        return (vocal, instrumental)
    }

    private func paddedChunk(_ x: [Float], start: Int, length: Int, valid: Int) -> [Float] {
        var out = [Float](repeating: 0, count: length)
        for i in 0..<valid { out[i] = x[start + i] }
        return out
    }

    // MARK: - File convenience

    /// Reads an audio/video file, separates it, and returns the stems.
    public func separate(contentsOf url: URL, progress: ((Double) -> Void)? = nil) throws -> Stems {
        let (channels, rate) = try Self.readAudio(url)
        return try separate(channels, sampleRate: rate, progress: progress)
    }

    // MARK: - Audio I/O helpers

    static func readAudio(_ url: URL) throws -> (channels: [[Float]], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let count = AVAudioFrameCount(file.length)
        guard count > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count) else {
            throw VocalIsolationError.audioReadFailed
        }
        try file.read(into: buffer)
        let frames = Int(buffer.frameLength)
        guard let data = buffer.floatChannelData else { throw VocalIsolationError.audioReadFailed }
        let channels = (0..<Int(format.channelCount)).map { c in
            Array(UnsafeBufferPointer(start: data[c], count: frames))
        }
        return (channels, format.sampleRate)
    }

    /// Writes stereo channels to a 32-bit float WAV.
    public static func writeWAV(_ channels: [[Float]], sampleRate: Double, to url: URL) throws {
        try? FileManager.default.removeItem(at: url)
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(channels.count), interleaved: false) else {
            throw VocalIsolationError.audioReadFailed
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        let frames = channels[0].count
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
            throw VocalIsolationError.audioReadFailed
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        for c in channels.indices {
            channels[c].withUnsafeBufferPointer { buffer.floatChannelData![c].update(from: $0.baseAddress!, count: frames) }
        }
        try file.write(from: buffer)
    }

    // MARK: - Resampling (AVAudioConverter)

    static func resample(_ input: [Float], from source: Double, to destination: Double) -> [Float] {
        guard source != destination, !input.isEmpty,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: destination, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(input.count))
        else { return input }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        inBuffer.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { inBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: input.count) }
        let capacity = AVAudioFrameCount(Double(input.count) * destination / source) + 16
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return input }
        let feeder = SingleBufferFeeder(inBuffer)
        let status = converter.convert(to: outBuffer, error: nil, withInputFrom: feeder.next)
        guard status != .error, let data = outBuffer.floatChannelData else { return input }
        return Array(UnsafeBufferPointer(start: data[0], count: Int(outBuffer.frameLength)))
    }

    private func matchLength(_ channels: [[Float]], to count: Int) -> [[Float]] {
        channels.map { ch in
            if ch.count == count { return ch }
            if ch.count > count { return Array(ch[0..<count]) }
            return ch + [Float](repeating: 0, count: count - ch.count)
        }
    }

    private final class SingleBufferFeeder: @unchecked Sendable {
        private let buffer: AVAudioPCMBuffer
        private var delivered = false
        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
        func next(_ count: AVAudioPacketCount, _ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
            if delivered { status.pointee = .endOfStream; return nil }
            delivered = true; status.pointee = .haveData; return buffer
        }
    }
}
