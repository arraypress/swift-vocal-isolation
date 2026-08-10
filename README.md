# VocalIsolation

On-device **vocal / instrumental separation** for Swift — split a song into a
clean vocal stem and its instrumental, entirely on-device (no cloud, no upload).
MDX-Net runs in Core ML; the STFT/ISTFT and chunked overlap-add run on
Accelerate/vDSP.

```swift
import VocalIsolation

let isolator = try VocalIsolator(modelURL: modelURL)
let stems = try isolator.separate(contentsOf: songURL)
try VocalIsolator.writeWAV(stems.vocal,        sampleRate: stems.sampleRate, to: vocalURL)
try VocalIsolator.writeWAV(stems.instrumental, sampleRate: stems.sampleRate, to: instURL)
```

## Two things it's for

- **Lyric captions** — feed `stems.vocal` to a speech recognizer (e.g. Whisper).
  On a dense mix the raw track gives a recognizer garbage; the isolated vocal
  gives real lyrics.
- **Acapella / producer tool** — keep the stems. `vocal` is the lead/backing
  vocal; `instrumental` is the residual (`mix − vocal`), so `vocal +
  instrumental == mix`.

## How it works

`VocalIsolator` operates on 44.1 kHz stereo (mono is duplicated; other rates are
resampled in and back). It processes the song in ~5.9 s chunks with 50 % overlap
and a Hann crossfade:

```
audio → STFT (vDSP, n_fft 8192) → MDX-Net (Core ML) → ISTFT (vDSP) → overlap-add
```

The STFT/ISTFT reproduce the model's PyTorch transform bit-for-bit (verified
against golden fixtures), and the full pipeline matches the reference within
noise. Roughly **5× faster than real time** on Apple Silicon (a 3–4 min song in
under a minute).

## The model

The MDX-Net model (MDX23C, MIT — credit [UVR](https://github.com/anjok07/ultimatevocalremovergui))
is **not bundled** — it's ~214 MB (fp16). Point `VocalIsolator(modelURL:)` at a
local `.mlpackage`/`.mlmodelc`, or fetch it once and cache it (like Whisper
weights). It's 44.1 kHz native, and outputs 2 stems (vocal, other).

**Quality note:** MDX-Net is strong on pop / hip-hop / EDM / most vocal-forward
music. Very dense rock/metal (walls of distorted guitar overlapping the vocal
band) is its ceiling — for a no-compromise result there, a Mel-Band Roformer is
higher quality, at a much larger model and a harder Core ML conversion.

## Requirements

macOS 26 / iOS 26, Swift 6.2. Core ML + Accelerate (system frameworks); no
external Swift dependencies. On-device — nothing leaves the device.

## Tests

```
swift test
```

The STFT/ISTFT tests run anywhere (bundled golden fixtures). The end-to-end
pipeline test is skipped unless a local model is present.

## License

MIT — see [LICENSE](LICENSE).

Bundled or downloaded models keep their own licences; see the notes above where a model is named.
