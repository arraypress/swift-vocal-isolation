"""Core ML vs PyTorch on the library's real audio fixture. Mask and audio both."""
import sys, warnings, yaml, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
import coremltools as ct
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import stft, apply_mask

FIX = "../Tests/VocalIsolationTests/Fixtures/e2e_chunk.bin"
raw = np.fromfile(FIX, dtype=np.float32)
audio_np = raw.reshape(2, -1)                       # [2, 261120] planar
print(f"fixture: {audio_np.shape}  peak={np.abs(audio_np).max():.4f} "
      f"rms={np.sqrt((audio_np**2).mean()):.4f}  L/R corr={np.corrcoef(audio_np)[0,1]:.3f}")

cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
chunk = cfg["audio"]["chunk_size"]
pad = np.zeros((2, chunk), np.float32)
n = min(chunk, audio_np.shape[1]); pad[:, :n] = audio_np[:, :n]
audio = torch.from_numpy(pad).unsqueeze(0)

mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
model = MelBandRoformer(**mcfg)
sd = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
model.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()})
model.eval()

with torch.no_grad():
    ref_audio = model(audio)
    spec = stft(audio, model)
    ref_mask = None

# PyTorch mask, for a like-for-like comparison with Core ML's output
from roformer_core import RoformerCore, staticize_rotary
core = RoformerCore(model, frames=spec.shape[2], num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
staticize_rotary(model, lambda: core(spec))
with torch.no_grad():
    ref_mask = core(spec)

ml = ct.models.MLModel("mel_band_roformer.mlpackage")
ni = list(ml.get_spec().description.input)[0].name
no = list(ml.get_spec().description.output)[0].name
ml_mask = torch.from_numpy(np.asarray(ml.predict({ni: spec.numpy().astype(np.float32)})[no])).float()

def cmp(name, a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = float(a @ b / (a.norm() * b.norm()))
    snr = 10 * torch.log10((a**2).sum() / ((a - b)**2).sum().clamp(min=1e-30))
    print(f"{name:18s} cosine={cos:.9f}  snr={snr:5.1f} dB  rms(ref)={a.pow(2).mean().sqrt():.5f}")

cmp("mask", ref_mask, ml_mask)
with torch.no_grad():
    ml_audio = apply_mask(spec, ml_mask, model, length=None)
k = min(ref_audio.shape[-1], ml_audio.shape[-1])
cmp("vocal audio", ref_audio[..., :k], ml_audio[..., :k])
print(f"vocal peak: ref={ref_audio.abs().max():.4f}  coreml={ml_audio.abs().max():.4f}")
