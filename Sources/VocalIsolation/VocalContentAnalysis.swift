//
//  VocalContentAnalysis.swift
//  VocalIsolation
//
//  Created by David Sherlock on 2026.
//

@preconcurrency import AVFoundation
import Foundation

/// What a track turns out to contain.
public enum VocalContent: String, Sendable {

    /// Both a vocal and a backing — an ordinary song. Separating it is worthwhile.
    case mixed

    /// No vocal worth extracting: a backing track, a karaoke version, or an already-separated
    /// instrumental. Asking for its vocal stem yields silence.
    case instrumentalOnly

    /// Vocal with essentially nothing behind it — an acapella, or an already-separated vocal
    /// stem. Asking for its instrumental yields silence.
    case acapella

    /// Too quiet to judge.
    case silent
}

/// The verdict plus the measurements behind it, so a caller can show its own message or apply
/// its own threshold.
public struct VocalContentAnalysis: Sendable {

    public let content: VocalContent

    /// Loudest vocal level found in any sampled window, in dB relative to that window's mix.
    /// `0` means the window is *all* vocal; very negative means there is no vocal in it.
    public let vocalLevelDB: Double

    /// Quietest instrumental level found, in dB relative to its window's mix.
    public let instrumentalLevelDB: Double

    /// How many windows were actually measured (silent ones are skipped).
    public let windowsMeasured: Int

    /// Public so a caller can construct a stand-in — e.g. assuming `.mixed` when the pre-pass
    /// itself fails, rather than failing the whole separation over a skipped optimisation.
    public init(
        content: VocalContent,
        vocalLevelDB: Double,
        instrumentalLevelDB: Double,
        windowsMeasured: Int
    ) {
        self.content = content
        self.vocalLevelDB = vocalLevelDB
        self.instrumentalLevelDB = instrumentalLevelDB
        self.windowsMeasured = windowsMeasured
    }
}

public extension VocalIsolator {

    /// Below this, a window has no vocal in it worth extracting.
    ///
    /// Calibrated against real material rather than guessed. Measured max-across-windows vocal
    /// level, in dB relative to the mix:
    ///
    /// | material                          | vocal |
    /// |-----------------------------------|-------|
    /// | commercial track, full mix        | −10.8 |
    /// | commercial track, full mix        |  −5.2 |
    /// | voice over a quiet bed            |  −0.2 |
    /// | *its own separated instrumental*  | −28.1 |
    /// | *its own separated instrumental*  | −39.5 |
    /// | pure instrumental bed             | −44.3 |
    ///
    /// The two classes are 17 dB apart at their closest (−10.8 vs −28.1). −20 sits in the gap
    /// with ~8 dB of margin on both sides.
    static let vocalPresenceFloorDB = -20.0

    /// Below this, a window has no backing behind the vocal.
    ///
    /// Same method. Measured min-across-windows instrumental level:
    /// acapellas came in at −32.5, −32.8 and −41.5; full mixes at −0.5, −1.9 and −12.6.
    /// The closest pair is −12.6 vs −32.5, so −25 leaves ~7 dB of margin.
    ///
    /// Note this floor is *lower* than the vocal one. A quiet backing is still a backing, and
    /// calling a sparse ballad an acapella would be the worse mistake — it would skip work the
    /// user asked for.
    static let instrumentalPresenceFloorDB = -25.0

    /// Work out whether a track is a normal song, an instrumental, or an acapella — **without
    /// separating the whole thing**.
    ///
    /// Separation runs at roughly 2.5x real time, so a four-minute track costs about 90 seconds.
    /// This samples a handful of single-chunk windows instead and costs 2–4 seconds, which is
    /// cheap enough to run before every separation. The point is not to be clever: it is that
    /// feeding in a karaoke track and getting back a silent "vocals" file is a confusing way to
    /// find out the track had no vocals in it.
    ///
    /// Windows are spread across the body of the track rather than taken from one place —
    /// sampling only the middle would call a song with a long instrumental break an
    /// instrumental, and sampling only the start would do the same to anything with an intro.
    /// Vocal presence therefore takes the **maximum** across windows: a track has vocals if
    /// *any* part of it does.
    ///
    /// - Parameters:
    ///   - url: the audio file to inspect.
    ///   - windows: how many places to sample. More is more robust on tracks with sparse vocals,
    ///     at roughly 0.6 s each.
    /// - Throws: `CancellationError` if the calling task is cancelled, or a `VocalIsolationError`.
    func analyseContent(contentsOf url: URL, windows: Int = 4) throws -> VocalContentAnalysis {
        let (channels, rate) = try Self.readAudio(url)
        return try analyseContent(channels, sampleRate: rate, windows: windows)
    }

    /// Analyse already-decoded channels. See `analyseContent(contentsOf:windows:)`.
    func analyseContent(
        _ channels: [[Float]],
        sampleRate: Double,
        windows: Int = 4
    ) throws -> VocalContentAnalysis {
        guard let first = channels.first, !first.isEmpty else { throw VocalIsolationError.emptyAudio }

        let total = first.count
        // One model chunk's worth, expressed at the file's own rate so each window costs exactly
        // one prediction after resampling.
        let windowLength = min(total, Int((Double(Self.chunk) * sampleRate / Self.modelRate).rounded()))
        let count = max(1, windows)

        var vocalDBs: [Double] = []
        var instrumentalDBs: [Double] = []

        for index in 0..<count {
            try Task.checkCancellation()

            // Centre of the i-th of `count` equal slices, so the samples are spread over the
            // whole track without ever running off either end.
            let fraction = (Double(index) + 0.5) / Double(count)
            let start = max(0, min(total - windowLength, Int(Double(total - windowLength) * fraction)))
            let slice = channels.map { Array($0[start..<(start + windowLength)]) }

            let mix = Self.rootMeanSquare(slice)
            // A silent window says nothing about the track — scoring it would drag a real
            // measurement toward whichever verdict silence happens to resemble.
            guard mix > 1e-6 else { continue }

            guard let stems = try? separate(slice, sampleRate: sampleRate) else { continue }
            vocalDBs.append(Self.decibels(Self.rootMeanSquare(stems.vocal), relativeTo: mix))
            instrumentalDBs.append(Self.decibels(Self.rootMeanSquare(stems.instrumental), relativeTo: mix))
        }

        guard let loudestVocal = vocalDBs.max(),
              let quietestInstrumental = instrumentalDBs.min() else {
            return VocalContentAnalysis(
                content: .silent,
                vocalLevelDB: -.infinity,
                instrumentalLevelDB: -.infinity,
                windowsMeasured: 0
            )
        }

        let content: VocalContent
        if loudestVocal < Self.vocalPresenceFloorDB {
            content = .instrumentalOnly
        } else if quietestInstrumental < Self.instrumentalPresenceFloorDB {
            content = .acapella
        } else {
            content = .mixed
        }

        return VocalContentAnalysis(
            content: content,
            vocalLevelDB: loudestVocal,
            instrumentalLevelDB: quietestInstrumental,
            windowsMeasured: vocalDBs.count
        )
    }

    // MARK: - Measurement

    static func rootMeanSquare(_ channels: [[Float]]) -> Double {
        var sum = 0.0
        var count = 0
        for channel in channels {
            for sample in channel { sum += Double(sample) * Double(sample) }
            count += channel.count
        }
        return count == 0 ? 0 : (sum / Double(count)).squareRoot()
    }

    static func decibels(_ value: Double, relativeTo reference: Double) -> Double {
        guard reference > 1e-12 else { return -.infinity }
        return 20 * log10(max(value, 1e-12) / reference)
    }
}
