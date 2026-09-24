# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "einops", "rotary-embedding-torch", "beartype", "ml_collections", "omegaconf", "pyyaml", "soundfile", "librosa"]
# ///
"""BS-RoFormer (Lu, Wang, Kong, Hung — ByteDance, ICASSP 2024; implementation lucidrains, MIT;
weights ZFTurbo, MIT) → stems-bs_roformer-float32.aimodel.

    uv run Tools/export_roformer.py --config <yaml> --checkpoint <ckpt> [--msst <checkout>] [--out Tools/exports] [--install]

The asset holds the network between the spectrogram and the mask at the model's chunk
(485,100 samples → 1,101 frames): input `x` [1, 1101, 4100] = the STFT laid out as
'b t (f s c)' — frequency bin, then channel, then real/imaginary — and output `mask`
[1, 4, 1101, 4100] in the same layout. STFT (n_fft 2048, hop 441, periodic Hann, not
normalised), the complex mask multiply, DC zeroing, inverse STFT and the chunked inference
with fades run in Swift (`RoFormerSTFT`, `RoFormerSeparator`), because Core AI has no STFT.
Everything else is upstream's forward verbatim, asserted equal to `model(chunk)` before export.
Needs a checkout of ZFTurbo/Music-Source-Separation-Training (`--msst`, `$MSST_DIR`, or
beside this repo)."""
import argparse, os, sys, time, shutil, json, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, soundfile as sf
from einops import rearrange, pack, unpack
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
ap.add_argument("--msst", default=os.environ.get("MSST_DIR") or str(HERE.parent.parent / "Music-Source-Separation-Training"))
ap.add_argument("--out", default=str(HERE / "exports")); ap.add_argument("--name", default="bs_roformer")
ap.add_argument("--install", action="store_true"); ap.add_argument("--clip", help="a wav to assert equality on (else noise)")
args = ap.parse_args()
sys.path.insert(0, args.msst); os.chdir(args.msst)
from utils.settings import get_model_from_config
import coreai_torch
from coreai.runtime import AIModelAssetMetadata

model, config = get_model_from_config("bs_roformer", args.config)
sd = torch.load(args.checkpoint, map_location="cpu")
for key in ("state", "state_dict"):
    if key in sd: sd = sd[key]
print("load:", model.load_state_dict(sd, strict=False), flush=True)
model.eval()
assert not model.skip_connection and all(len(b) == 2 for b in model.layers), "only the plain time/freq block layout is exported"
L = int(config.audio.chunk_size); hop = model.stft_kwargs["hop_length"]; nfft = model.stft_kwargs["n_fft"]
T = 1 + L // hop; F = nfft // 2 + 1; S_ = model.audio_channels; N = len(model.mask_estimators)
print(f"{args.name}: chunk {L}, frames {T}, bins {F}, channels {S_}, stems {N}, dim {config.model.dim} depth {config.model.depth}, params {sum(p.numel() for p in model.parameters())/1e6:.1f} M, zero_dc {model.zero_dc}", flush=True)

class Core(nn.Module):
    """Upstream forward from the band-split input to the mask, verbatim."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):                                   # [b, t, (f s c)]
        m = self.m
        x = m.band_split(x)
        for time_transformer, freq_transformer in m.layers:
            x = rearrange(x, 'b t f d -> b f t d'); x, ps = pack([x], '* t d'); x = time_transformer(x); x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d'); x, ps = pack([x], '* f d'); x = freq_transformer(x); x, = unpack(x, ps, '* f d')
        x = m.final_norm(x)
        return torch.stack([fn(x) for fn in m.mask_estimators], dim=1)   # [b, n, t, (f c)]

core = Core(model).eval()
def spectrogram(chunk):                                     # chunk [1, 2, L] → x [1, T, (f s c)], and the complex STFT
    window = model.stft_window_fn(device=chunk.device)
    raw, shape = pack([chunk], '* t')
    rep = torch.stft(raw, **model.stft_kwargs, window=window, return_complex=True)
    rep = torch.view_as_real(rep).view(1, S_, F, T, 2)              # unpack_one(…, '* f t c')
    rep = rearrange(rep, 'b s f t c -> b (f s) t c')
    return rearrange(rep, 'b f t c -> b t (f c)'), rep
def finish(mask, rep, length):                               # upstream's tail: complex mask, zero DC, istft
    window = model.stft_window_fn(device=mask.device)
    mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)
    z = torch.view_as_complex(rep.unsqueeze(1).contiguous()) * torch.view_as_complex(mask.contiguous())
    z = rearrange(z, 'b n (f s) t -> (b n s) f t', s=S_)
    if model.zero_dc: z[:, 0] = 0.
    audio = torch.istft(z, **model.stft_kwargs, window=window, return_complex=False, length=length)
    return rearrange(audio, '(b n s) t -> b n s t', s=S_, n=N)

if args.clip:
    y, sr = sf.read(args.clip, dtype='float32', always_2d=True); y = y.T
    chunk = torch.from_numpy(np.ascontiguousarray(y[:2, :L]))[None]
    if chunk.shape[-1] < L: chunk = torch.nn.functional.pad(chunk, (0, L - chunk.shape[-1]), mode='reflect')
else:
    chunk = torch.randn(1, 2, L) * 0.1
with torch.no_grad():
    x, rep = spectrogram(chunk)
    ours = finish(core(x), rep, L); theirs = model(chunk)
print(f"Core + STFT/mask/iSTFT vs model(chunk): max |diff| {(ours - theirs).abs().max():.2e} (equal: {torch.equal(ours, theirs)})", flush=True)
assert torch.equal(ours, theirs)

t0 = time.time()
with torch.no_grad():
    ex = torch.export.export(core, (x,)).run_decompositions(coreai_torch.get_decomp_table())
ops = sorted({str(n.target) for n in ex.graph.nodes if n.op == 'call_function'})
print(f"exported in {time.time()-t0:.0f} s: {sum(1 for n in ex.graph.nodes if n.op=='call_function')} ops; kinds: {len(ops)}", flush=True)
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex, input_names=['x'], output_names=['mask'], entrypoint_name='main')
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Lu, Wang, Kong, Hung (ByteDance) — BS-RoFormer; implementation lucidrains (MIT); weights ZFTurbo (MIT); Core AI export by VocalIsolation"
meta.license = "MIT"
meta.model_description = f"BS-RoFormer {args.name}: x [1,{T},{2*S_*F}] = STFT as (f s c) per frame → mask [1,{N},{T},{2*S_*F}]; STFT (n_fft {nfft}, hop {hop}), complex mask, DC zeroing, iSTFT and chunking in the host. Stems {list(config.training.instruments)}."
meta.creation_date = int(time.time())
out = Path(args.out) / f"stems-{args.name}-float32.aimodel"; out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): shutil.rmtree(out)
program.save_asset(out, meta)
print(f"saved {out} ({sum(f.stat().st_size for f in out.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/stems" / out.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(out, dest); print(f"installed {dest}")
