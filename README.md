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

## Four stems: Hybrid Transformer Demucs

`DemucsSeparator` runs Meta's HTDemucs (`htdemucs`, MIT, 42M parameters) on Core AI and returns
drums, bass, other and vocals. The MDX model above stays the better vocal/instrumental split;
Demucs is for the stems it cannot give — a bass or drum stem for transcription, for instance.

```swift
let separator = try await DemucsSeparator(contentsOf: assetURL)      // stems-htdemucs-float32.aimodel
let stems = try await separator.separate(channels, sampleRate: 44_100)
stems["bass"]   // [[Float]] (L, R), in DemucsSeparator.sources order under stems.stems
```

**How it is applied is upstream's, line for line.** Core AI has no STFT and no variance op, so
the asset holds the network between the complex spectrogram and the mask at the model's
7.8-second training segment, and Swift does the rest exactly as `demucs.api.Separator` and
`apply_model` do: the global normalisation by the mono mix, chunks at a 75% stride, the
triangular overlap-add weights, the centred padding of the last chunk and its centre trim, and
the per-segment statistics the model computes on its input. Random shifts (`shifts=1` upstream)
are off: they make upstream itself non-deterministic, and parity needs a fixed answer.

Held to upstream's Python (`Tools/export_demucs.py` writes the references, `DemucsTests` reads
them from `STEMS_DEMUCS_REFS`):

| stage | result |
|---|---|
| the network from spectrogram to mask | asserted equal to `model(mix)` before export |
| STFT and inverse STFT | 154 dB PSNR against torch |
| one training segment through Core AI (GPU) | 136–147 dB against upstream's output |
| every one of 63 chunks of a 6-minute mix | worst stem 110 dB |
| whole clips end to end, 10 s and 6 min | 127–148 dB on every stem |

So the stems are upstream's to the float noise floor. The asset is published at
[huggingface.co/arraypress/stems-demucs](https://huggingface.co/arraypress/stems-demucs) (168 MB);
`uv run Tools/export_demucs.py --install` rebuilds it from the upstream checkpoint.

**The fine-tuned bag, `htdemucs_ft`.** Four models, one fine-tuned per stem, about one dB better
on MUSDB and four times the compute. The same code path: `stems-htdemucs_ft-float32.aimodel`
(673 MB, same repo) carries four entry points and `DemucsSeparator` takes stem k from model k, as
upstream's one-hot bag weights do. Each of the four is asserted equal to upstream before export;
segments 140–147 dB, whole clips 135–147 dB on every stem. The 6-minute mix takes 127 s verified
(2.9× realtime) against 73 s for `htdemucs`.

## Four stems, best quality: BS-RoFormer

`RoFormerSeparator` runs BS-RoFormer (Lu, Wang, Kong, Hung — ByteDance, ICASSP 2024; lucidrains'
implementation, MIT) with the four-stem weights ZFTurbo trained and released under MIT: MUSDB18
test average **9.65 dB** SDR (drums 11.6, vocals 11.1, bass 8.5, other 7.4), against about 9.0 for
`htdemucs_ft` and 8.0 for `htdemucs`. 132M parameters. In the CLI: `--engine roformer`.

The band-split transformer lives in `stems-bs_roformer-float32.aimodel` (528 MB) at the model's
485,100-sample chunk; the STFT (n_fft 2048, hop 441, periodic Hann, unnormalised, laid out per
frame as bin, channel, real/imaginary), the complex mask, the DC zeroing, the inverse and
upstream's `demix` — 50% step, reflect border, linear fades dropped on the first and last batch,
sum over count — run in Swift. The batches are replayed as upstream forms them, because the fade
rule is per batch.

| stage | result |
|---|---|
| the network from spectrogram to mask | asserted equal to `model(chunk)` before export |
| spectrogram layout, masked inverse | 155 dB, 151 dB against torch |
| each of 72 chunks through Core AI (GPU), 10 s and 6 min | 74–110 dB against upstream |
| whole clips end to end, 10 s and 6 min | 97–114 dB on every stem |

Lower than the convolutional models' 140 dB because eight layers of fp32 attention accumulate
GPU-versus-CPU rounding — a relative error of 1e-4 on the worst chunk, 1e-5 typical, far below
anything audible, and repeatable run to run. The same run-twice verification applies. Asset at
[huggingface.co/arraypress/stems-roformer](https://huggingface.co/arraypress/stems-roformer);
`Tools/export_roformer.py` rebuilds it from ZFTurbo's config and checkpoint.

## Requirements

macOS 27 (Core AI for Demucs; the MDX path alone needed only 26), Swift 6.2. Core ML, Core AI and Accelerate (system frameworks); no
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
