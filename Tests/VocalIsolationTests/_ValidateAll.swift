//  TEMPORARY — runs the shipped VocalIsolator on every test song and writes
//  <prefix>_swift_vocal.wav to Downloads for comparison. Delete before publish.

import Foundation
import Testing
@testable import VocalIsolation

@Suite("ValidateAll", .serialized)
struct ValidateAll {

    private var localModelURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("mdx_net.mlpackage")
    }

    @Test("Separate all test songs with the shipped library")
    func all() throws {
        guard FileManager.default.fileExists(atPath: localModelURL.path) else { print("[skip] no model"); return }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dl = home.appending(path: "Downloads")
        let songs: [(String, String)] = [
            ("Amber - Sexual (Li Da Di) (Original Radio Edit).wav", "amber"),
            ("Sharon Little - Don't Mash Up Creation - 01 Don't Mash Up Creation.wav", "sharon"),
            ("Calvin-Harris-feat-Jazzy-Satisfy-(SixLoaded.com).mp3", "calvin"),
            ("Wisp - If not winter.wav", "wisp"),
            ("Evanescence - Lithium.wav", "lithium"),
            ("vocalstacktest.wav", "vocalstack"),
        ]
        let isolator = try VocalIsolator(modelURL: localModelURL)
        for (file, prefix) in songs {
            let url = dl.appending(path: file)
            guard FileManager.default.fileExists(atPath: url.path) else { print("SWIFT \(prefix): missing"); continue }
            let clock = Date()
            let stems = try isolator.separate(contentsOf: url)
            try VocalIsolator.writeWAV(stems.vocal, sampleRate: stems.sampleRate, to: dl.appending(path: "\(prefix)_swift_vocal.wav"))
            try VocalIsolator.writeWAV(stems.instrumental, sampleRate: stems.sampleRate, to: dl.appending(path: "\(prefix)_swift_instrumental.wav"))
            print("SWIFT \(prefix): \(stems.vocal[0].count / Int(stems.sampleRate))s @ \(Int(stems.sampleRate))Hz in \(Int(-clock.timeIntervalSinceNow))s", terminator: "\n")
            fflush(stdout)
        }
        print("SWIFT ALL DONE")
    }
}
