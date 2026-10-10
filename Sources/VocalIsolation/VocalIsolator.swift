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
    /// No model at the path given (Demucs).
    case modelNotFound(String)
    /// The Core AI asset would not load, lacks its entry point, or returned nothing (Demucs).
    case modelUnavailable(String)
    case modelLoadFailed
    case dspSetupFailed
    case predictionFailed
    case audioReadFailed
    case emptyAudio
}

/// On-device vocal / instrumental separation with MDX-Net (MDX23C).
///
/// Splits a song into a clean **vocal** stem and its **instrumental** — feed the
/// vocal to a speech recognizer for lyric captions, or keep the stems as an
/// acapella / producer tool. Everything runs on-device; the model is 44.1 kHz
/// stereo, and audio at other rates/layouts is converted in and out.
///
/// This is the original 8-bit Core ML package. The same network at full
/// precision, as a Core AI asset, is ``MDXSeparator`` — measured 124 dB to torch
/// against this package's 35 dB, and 2.4× faster a chunk.
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
        let stereo = try Self.stereoAtModelRate(channels, sampleRate: sampleRate)
        let (vocal, instrumental) = try separateAt44k(stereo, progress: progress)
        return Self.fromModelRate(vocal: vocal, instrumental: instrumental, sampleRate: sampleRate, length: channels[0].count)
    }

    /// Stereo at the model's rate: mono duplicated, any rate resampled.
    static func stereoAtModelRate(_ channels: [[Float]], sampleRate: Double) throws -> [[Float]] {
        guard let first = channels.first, !first.isEmpty else { throw VocalIsolationError.emptyAudio }
        let stereo = channels.count >= 2 ? [channels[0], channels[1]] : [channels[0], channels[0]]
        guard sampleRate != modelRate else { return stereo }
        return stereo.map { resample($0, from: sampleRate, to: modelRate) }
    }

    /// The stems back at the input's rate and length.
    static func fromModelRate(vocal: [[Float]], instrumental: [[Float]], sampleRate: Double, length: Int) -> Stems {
        guard sampleRate != modelRate else { return Stems(vocal: vocal, instrumental: instrumental, sampleRate: sampleRate) }
        let back = { (x: [[Float]]) in x.map { resample($0, from: modelRate, to: sampleRate) } }
        return Stems(vocal: matchLength(back(vocal), to: length), instrumental: matchLength(back(instrumental), to: length), sampleRate: sampleRate)
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
        var plan = ChunkPlan(stereo: stereo)
        for (index, start) in plan.starts.enumerated() {
            try Task.checkCancellation()
            let (spec, frames, valid) = Self.chunkSpectrogram(stereo, start: start, stft: stft)
            // A dead chunk = silence in the output; still advance the
            // progress bar so a rare failure doesn't look like a hang.
            if let stems = try? model.predict(spec) { plan.add(stems, frames: frames, start: start, valid: valid, stft: stft) }
            progress?(Double(index + 1) / Double(plan.starts.count))
        }
        return plan.finish(stereo)
    }

    /// One chunk's spectrogram, zero-padded past the end of the recording.
    static func chunkSpectrogram(_ stereo: [[Float]], start: Int, stft: VocalSTFT) -> (spec: [Float], frames: Int, valid: Int) {
        let valid = min(chunk, stereo[0].count - start)
        let left = paddedChunk(stereo[0], start: start, length: chunk, valid: valid)
        let right = paddedChunk(stereo[1], start: start, length: chunk, valid: valid)
        let (spec, frames) = stft.forward([left, right])
        return (spec, frames, valid)
    }

    /// The overlap-add of the vocal across 50 %-overlapping, Hann-weighted
    /// chunks; the instrumental is what the vocal leaves of the mix.
    struct ChunkPlan {
        let starts: [Int]
        private let window: [Float]
        private var vocal: [[Float]]
        private var weight: [Float]

        init(stereo: [[Float]]) {
            let C = VocalIsolator.chunk, hop = C / VocalIsolator.overlap, n = stereo[0].count
            window = (0..<C).map { 0.5 - 0.5 * cos(2 * .pi * Float($0) / Float(C)) }
            vocal = [[Float]](repeating: [Float](repeating: 0, count: n), count: 2)
            weight = [Float](repeating: 0, count: n)
            var starts = Array(stride(from: 0, through: max(0, n - C), by: hop))
            if let last = starts.last, last + C < n { starts.append(n - C) }
            self.starts = starts.isEmpty ? [0] : starts
        }

        mutating func add(_ stems: [Float], frames: Int, start: Int, valid: Int, stft: VocalSTFT) {
            let stemSize = VocalModel.specChannels * VocalModel.dimF * VocalModel.frames
            let chunk = stft.inverse(Array(stems[0..<stemSize]), frames: frames)   // stem 0 = vocal
            for i in 0..<valid {
                let w = window[i]
                vocal[0][start + i] += chunk[0][i] * w
                vocal[1][start + i] += chunk[1][i] * w
                weight[start + i] += w
            }
        }

        func finish(_ stereo: [[Float]]) -> (vocal: [[Float]], instrumental: [[Float]]) {
            var vocal = self.vocal
            var instrumental = vocal
            for ch in 0..<2 {
                for i in 0..<stereo[ch].count {
                    let v = weight[i] > 1e-6 ? vocal[ch][i] / weight[i] : 0
                    vocal[ch][i] = v
                    instrumental[ch][i] = stereo[ch][i] - v    // residual = everything but the vocal
                }
            }
            return (vocal, instrumental)
        }
    }

    private static func paddedChunk(_ x: [Float], start: Int, length: Int, valid: Int) -> [Float] {
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

    private static func matchLength(_ channels: [[Float]], to count: Int) -> [[Float]] {
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
