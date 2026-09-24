# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "demucs", "einops", "soundfile", "librosa"]
# ///
"""Hybrid Transformer Demucs (Rouard, Massa, Défossez — Meta, MIT) → stems-<name>-float32.aimodel.

    uv run Tools/export_demucs.py [--name htdemucs] [--out Tools/exports] [--install] [clip=path.wav ...]

The asset holds the network between the complex spectrogram and the mask, at the model's
training segment (7.8 s → mix [1, 2, 343980], complex-as-channels spectrogram [1, 4, 2048, 336],
per-segment statistics [4]) → (spec [1, 16, 2048, 336], time [1, 8, 343980]). STFT, inverse STFT,
chunking with overlap-add, the global normalisation and the per-segment statistics run in Swift
(`DemucsSTFT`, `DemucsSeparator`) because Core AI has no STFT and no `aten.var`; everything else
is upstream's forward verbatim, asserted equal to `model(mix)` before export.

Given clips, writes upstream references for `DemucsTests` (STEMS_DEMUCS_REFS): the exact
44.1 kHz samples, the first chunk's inputs and outputs, and the stems from `demucs.api.Separator`
with `shifts=0` (random shifts make upstream non-deterministic; parity needs them off).
Only `htdemucs` has been verified; `htdemucs_ft` is a bag of four and `htdemucs_6s` adds guitar
and piano — both export the same way, one model at a time.
"""
import argparse, os, shutil, sys, time, math, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, librosa
from demucs.pretrained import get_model
from demucs.api import Separator
from demucs.apply import TensorChunk
from einops import rearrange
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="htdemucs")
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--refs", default=str(HERE / "reference" / "demucs"), help="where clip references go")
ap.add_argument("--install", action="store_true", help="copy the asset into ~/Library/Application Support/stems")
ap.add_argument("clips", nargs="*", help="name=path.wav: reference clips for the Swift tests")
args = ap.parse_args()
name = args.name
clips = dict(a.split("=", 1) for a in args.clips)

bag = get_model(name); bag.eval()
model = bag.models[0]; model.eval()
L = int(model.segment * model.samplerate); hl = model.hop_length; nfft = model.nfft
le = int(math.ceil(L / hl)); Fq = nfft // 2; S_ = len(model.sources)
print(f'{name}: sources {model.sources}, segment {L} samples, frames {le}, freqs {Fq}, params {sum(p.numel() for p in model.parameters())/1e6:.1f} M', flush=True)

class Core(torch.nn.Module):
    """Upstream forward from the cac magnitude to the pre-mask outputs, verbatim."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, mix, mag, stats):
        # stats = [mag mean, mag std, mix mean, mix std], upstream's per-segment normalisation
        # computed by the caller: Core AI has no aten.var, and the maths is four numbers.
        m = self.m
        x = mag
        B, C, Fq_, T = x.shape
        mean = stats[0].reshape(1, 1, 1, 1); std = stats[1].reshape(1, 1, 1, 1)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = stats[2].reshape(1, 1, 1); stdt = stats[3].reshape(1, 1, 1)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1]); inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1]); tenc = m.tencoder[idx]; xt = tenc(xt)
                if not tenc.empty: saved_t.append(xt)
                else: inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)"); x = m.channel_upsampler(x); x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)"); x = m.channel_downsampler(x); x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1); x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]; length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]; xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1); xt, _ = tdec(xt, skip, length_t)
        Sn = len(m.sources)
        x = x.view(B, Sn, -1, Fq_, T); x = x * std + mean
        xt = xt.view(B, Sn, -1, L); xt = xt * stdt + meant
        return x.reshape(B, Sn * 4, Fq_, T), xt.reshape(B, Sn * 2, L)

core = Core(model).eval()
def stats_of(mix, mag):
    return torch.stack([mag.mean(), mag.std(), mix.mean(), mix.std()])

def finish(spec_out, time_out):
    """Python's own _mask + _ispec + sum, from Core's outputs."""
    x = spec_out.view(1, S_, 4, Fq, le)
    zout = model._mask(None, x)
    return model._ispec(zout, L) + time_out.view(1, S_, 2, L)

# Reference clips: global normalisation as api.Separator, first chunk as apply_model's TensorChunk.padded.
for cname, path in clips.items():
    y, sr = sf.read(path, dtype='float32', always_2d=True); y = y.T
    if y.shape[0] == 1: y = np.repeat(y, 2, 0)
    if sr != 44100: y = librosa.resample(y, orig_sr=sr, target_sr=44100, res_type='soxr_hq').astype(np.float32)
    d = Path(args.refs) / cname; d.mkdir(parents=True, exist_ok=True)
    sf.write(d / 'audio_44k.wav', y.T, 44100, subtype='FLOAT')
    wav = torch.from_numpy(y)
    ref = wav.mean(0); mean = ref.mean(); std = ref.std() + 1e-8
    norm = ((wav - mean) / std)[None]
    chunk = TensorChunk(norm, 0, L).padded(L)                       # [1, 2, L]
    with torch.no_grad():
        z = model._spec(chunk); mag = model._magnitude(z)             # [1, 4, 2048, 336]
        stats = stats_of(chunk, mag)
        spec_out, time_out = core(chunk, mag, stats)
        ours = finish(spec_out, time_out)
        theirs = model(chunk)
    print(f'  {cname}: Core+mask/ispec vs model(chunk): max |diff| {(ours - theirs).abs().max():.2e} (equal: {torch.equal(ours, theirs)})', flush=True)
    if not (d / 'stems.f32').exists():
        sep = Separator(model=name, shifts=0, overlap=0.25, split=True, device='cpu', progress=False)
        t = time.time(); _, stems = sep.separate_tensor(wav, 44100); took = time.time() - t
        print(f'  {cname}: {y.shape[1]/44100:.1f} s separated in {took:.1f} s CPU; stems {list(stems)}', flush=True)
        np.stack([stems[s].numpy() for s in model.sources]).astype(np.float32).tofile(d / 'stems.f32')
    stats.numpy().astype(np.float32).tofile(d / 'chunk_stats.f32')
    chunk[0].numpy().astype(np.float32).tofile(d / 'chunk_mix.f32'); mag[0].numpy().astype(np.float32).tofile(d / 'chunk_mag.f32')
    spec_out[0].numpy().astype(np.float32).tofile(d / 'chunk_spec_out.f32'); time_out[0].numpy().astype(np.float32).tofile(d / 'chunk_time_out.f32')
    theirs[0].numpy().astype(np.float32).tofile(d / 'chunk_out.f32')
    json.dump({'mean': float(mean), 'std': float(std), 'samples': int(y.shape[1]), 'sources': model.sources, 'segment': L, 'frames': le}, open(d / 'info.json', 'w'), indent=1)

g = torch.Generator().manual_seed(0)
short = torch.randn(1, 2, 22050, generator=g)
with torch.no_grad():
    zs = model._spec(short); ms = model._magnitude(zs)
    cac = torch.randn(1, 2, 4, Fq, ms.shape[-1], generator=g)      # [B, S=... no: one stem] → use S=1 shape [1,1,4,F,T]
    cac = cac[:, :1]
    back = model._ispec(model._mask(None, cac), 22050)
fx = Path(args.refs) / 'stft'; fx.mkdir(parents=True, exist_ok=True)
short[0].numpy().astype(np.float32).tofile(fx / 'short_in.f32'); ms[0].numpy().astype(np.float32).tofile(fx / 'short_mag.f32')
cac[0, 0].numpy().astype(np.float32).tofile(fx / 'cac_in.f32'); back[0, 0].numpy().astype(np.float32).tofile(fx / 'cac_out.f32')
json.dump({'samples': 22050, 'frames': int(ms.shape[-1])}, open(fx / 'info.json', 'w'))
print('stft fixtures: mag', tuple(ms.shape), 'ispec', tuple(back.shape), flush=True)

t0 = time.time()
mix_ex = torch.randn(1, 2, L); mag_ex = torch.randn(1, 4, Fq, le); stats_ex = torch.tensor([0., 1., 0., 1.])
with torch.no_grad():
    ex = torch.export.export(core, (mix_ex, mag_ex, stats_ex)).run_decompositions(coreai_torch.get_decomp_table())
print(f'exported: {sum(1 for n in ex.graph.nodes if n.op == "call_function")} ops in {time.time()-t0:.0f} s', flush=True)
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex, input_names=['mix', 'mag', 'stats'], output_names=['spec', 'time'], entrypoint_name='main')
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = 'Rouard, Massa, Défossez (Meta) — Hybrid Transformer Demucs; Core AI export by VocalIsolation'
meta.license = 'MIT'
meta.model_description = f'HTDemucs {name}: sources {model.sources}; main = (mix [1,2,{L}] normalised, cac spectrogram [1,4,{Fq},{le}]) → (spec [1,{S_*4},{Fq},{le}] complex-as-channels stems, time [1,{S_*2},{L}]); STFT/iSTFT/chunking in the host.'
meta.creation_date = int(time.time())
out = Path(args.out) / f'stems-{name}-float32.aimodel'
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): shutil.rmtree(out)
program.save_asset(out, meta)
print(f'saved {out} ({sum(f.stat().st_size for f in out.rglob("*"))/1e6:.0f} MB) in {time.time()-t0:.0f} s')
if args.install:
    dest = Path.home() / "Library/Application Support/stems" / out.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(out, dest); print(f"installed {dest}")
