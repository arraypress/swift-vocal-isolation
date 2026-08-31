"""Split pipeline (host STFT -> core -> mask -> host ISTFT) vs the stock model."""
import sys, warnings, yaml, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "_ref/mss")
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
torch.manual_seed(0)
audio = (torch.randn(1, 2, chunk) * 0.05).clamp(-1, 1)

with torch.no_grad():
    ref = model(audio)
    spec = stft(audio, model)
    core = RoformerCore(model, frames=spec.shape[2],
                        num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
    mask = core(spec)
    ours = apply_mask(spec, mask, model, length=None)

n = min(ref.shape[-1], ours.shape[-1])
a, b = ref[..., :n].double().flatten(), ours[..., :n].double().flatten()
cos = float(a @ b / (a.norm() * b.norm()))
snr = 10 * torch.log10((a**2).sum() / ((a - b)**2).sum().clamp(min=1e-30))
print(f"spec {tuple(spec.shape)} mask {tuple(mask.shape)}")
print(f"cosine={cos:.12f}  snr={snr:.1f} dB  maxdiff={(a-b).abs().max():.3e}")
