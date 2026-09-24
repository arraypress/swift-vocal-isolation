// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VocalIsolation",
    platforms: [
        .macOS("27.0"),
        .iOS(.v26),
    ],
    products: [
        // On-device vocal / instrumental separation (MDX-Net via Core ML).
        // Split a song into a clean vocal stem + instrumental — for lyric
        // transcription (feed the vocal to a recognizer) or as an acapella /
        // producer tool. The model is fetched on first use, not bundled.
        .library(name: "VocalIsolation", targets: ["VocalIsolation"]),
    ],
    targets: [
        .target(name: "VocalIsolation"),
        .testTarget(
            name: "VocalIsolationTests",
            dependencies: ["VocalIsolation"],
            resources: [
                .copy("Fixtures/Demucs"),
                .copy("Fixtures/stft_audio.bin"),
                .copy("Fixtures/stft_spec.bin"),
                .copy("Fixtures/stft_recon.bin"),
                .copy("Fixtures/e2e_chunk.bin"),
                .copy("Fixtures/e2e_vocal.bin"),
                .copy("Fixtures/dereverb_chunk.bin"),
                .copy("Fixtures/dereverb_dry.bin"),
            ]
        ),
    ]
)
