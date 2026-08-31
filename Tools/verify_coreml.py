"""Core ML output vs stock PyTorch, on real-ish audio, measured end to end."""
import sys, warnings, yaml, numpy as np, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
import coremltools as ct
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import RoformerCore, stft, apply_mask

cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
model = MelBandRoformer(**mcfg)
sd = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
model.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()})
model.eval()

chunk = cfg["audio"]["chunk_size"]
sr = cfg["audio"]["sample_rate"]
# A tone-plus-noise mix: exercises real spectral structure, not just white noise.
t = torch.arange(chunk, dtype=torch.float32) / sr
tone = 0.25 * torch.sin(2 * torch.pi * 220 * t) + 0.15 * torch.sin(2 * torch.pi * 660 * t)
torch.manual_seed(0)
audio = (tone.unsqueeze(0).repeat(2, 1) + 0.05 * torch.randn(2, chunk)).unsqueeze(0).clamp(-1, 1)

with torch.no_grad():
    ref = model(audio)
    spec = stft(audio, model)

ml = ct.models.MLModel("mel_band_roformer.mlpackage")
name_in = list(ml.get_spec().description.input)[0].name
name_out = list(ml.get_spec().description.output)[0].name
pred = ml.predict({name_in: spec.numpy().astype(np.float32)})[name_out]
mask = torch.from_numpy(np.asarray(pred)).float()
with torch.no_grad():
    ours = apply_mask(spec, mask, model, length=None)

k = min(ref.shape[-1], ours.shape[-1])
a, b = ref[..., :k].double().flatten(), ours[..., :k].double().flatten()
cos = float(a @ b / (a.norm() * b.norm()))
snr = 10 * torch.log10((a**2).sum() / ((a - b)**2).sum().clamp(min=1e-30))
print(f"io: {name_in} -> {name_out}")
print(f"vocal energy: ref rms={a.pow(2).mean().sqrt():.5f}  coreml rms={b.pow(2).mean().sqrt():.5f}")
print(f"cosine={cos:.9f}  snr={snr:.1f} dB  maxdiff={(a-b).abs().max():.3e}")
