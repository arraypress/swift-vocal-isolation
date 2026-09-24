# CLAUDE.md — swift-vocal-isolation

Two separators. `VocalIsolator`: MDX-Net vocal/instrumental via Core ML (the
original, measured better on vocals). `DemucsSeparator`: HTDemucs four stems
via Core AI (added 2026-09-24). Module `VocalIsolation`; the CLI is
`../swift-stems-cli` (binary `stems`). What a re-derivation would get wrong:

## Build & test
```bash
swift build && swift test                          # STFT pairs and small fixtures, no model
STEMS_DEMUCS_REFS=<refs dir> swift test --filter DemucsTests   # + one segment, every chunk, whole clips vs upstream
```
Model-backed Demucs tests need the asset installed at
`~/Library/Application Support/stems/stems-htdemucs-float32.aimodel` and a
references folder from `uv run Tools/export_demucs.py demo=clip.wav …`.
The package floor is macOS 27 (Core AI); MDX via Core ML still works there.

## Demucs traps (all measured)
- **Core AI has no STFT and no `aten.var`.** The export takes the network
  between `_magnitude` and `_mask` at the fixed training segment (7.8 s =
  343,980 samples, 336 frames) and passes the four per-segment statistics
  (mag mean/std, mix mean/std, unbiased) as an input. STFT/iSTFT are
  `DemucsSTFT`: `_spec` reflect-pads 3·hop/2 left and to a hop multiple
  right, drops Nyquist, trims two frames each end; `_ispec` reverses it with
  zero Nyquist and two zero frames each side; `normalized=True` is ×1/√N
  forward and ×√N inverse. 154 dB vs torch.
- **Chunking is `apply_model` exactly**: stride 257,985 (75%), triangular
  weight `arange(1..L/2) ++ arange(L−L/2..1)` / max, and the LAST chunk is
  CENTRED-padded by `TensorChunk.padded` (real audio from BEFORE the offset
  fills the front, zeros only past the ends) then `center_trim`med — verified
  empirically with a forward hook, not from reading the code, because the
  `segment`/`valid_length` branch in `apply_model` is easy to misread.
- **Shifts are off.** Upstream's default `shifts=1` adds a random time shift;
  parity is impossible with it on. `Separator(shifts=0)` for references.
- **The reassembly invariant does not hold for Demucs**: stems sum to the mix
  the model *heard*, not the input, so do not port the MDX complement test.
- **The GPU is intermittently wrong, ~2% of segments.** Measured with
  `faultRate` (STEMS_DEMUCS_PASSES=6): 7 of 378 segment runs came back at
  77–105 dB against upstream where a correct one is 136–147 dB; which chunk
  varies run to run, the same chunk alone is fine ×4, and neither a pause
  before reading the output nor owning the output buffers (`outputViews`)
  changed the rate. Core AI's CPU path segfaults on this graph. A correct
  run IS bit-for-bit repeatable, so `DemucsSeparator(verifying: true)` (the
  default) runs each segment twice and arbitrates a mismatch with a third
  run — `recomputedSegments` counts them. Model time is only 0.7 s of the
  2.3 s per segment (STFT 0.25 s, iSTFT 4 × 0.34 s before caching the
  envelope), so verification costs ~30%.
- **Two Core AI jobs on the GPU at once break verification.** With the
  RoFormer parity test running in another process, both Demucs engines on a
  3-minute track threw "three runs disagreed" — under contention runs stop
  agreeing at all. Verification now tries up to 5 runs and reports the
  pairwise PSNR spread in the error (float noise ≥120 dB = not repeatable
  under load; 60–110 dB = the wrong-segment fault). Do not run two
  separations concurrently; queue them.
- **`htdemucs_ft`**: a `BagOfModels` with ONE-HOT weights — stem k comes from
  model k, so the asset has `model0…model3` entry points and the separator
  runs all four per segment (verification per model run). Same STFT and
  chunking. 2.9× realtime verified on the release CLI.
- **BS-RoFormer** (`RoFormerSeparator`, `RoFormerSTFT`): n_fft 2048, hop 441,
  NOT normalised, Nyquist kept, DC zeroed after the mask (`zero_dc`); the
  model input is 'b t (f s c)' — bin-major, then channel, then re/im —
  and the mask comes back in the same layout. `demix` generic mode:
  chunk 485,100, `num_overlap` 2 (step 242,550), fade chunk/10, border =
  chunk − step reflect-padded when length > 2·border, chunks shorter than
  half a chunk are zero-padded (else reflect), the fade rule is decided PER
  BATCH of `batch_size` 2 (first batch keeps fade-in only if it holds one
  chunk; the last batch drops the fade-out), `result/counter` with NaN → 0.
  Config values are read from the YAML, not the model defaults (defaults say
  hop 512, overlap 4). Core AI converted the transformer as is (RoPE, SDPA,
  RMSNorm via F.normalize, GLU) — no re-authoring; 96–110 dB per chunk is
  what eight layers of fp32 attention on the GPU give.
- Test fixtures under `Fixtures/Demucs/` are reached with
  `Bundle.module.url(..., subdirectory: "Demucs")` — the `.copy` keeps the
  folder name, not the `Fixtures/` prefix.
