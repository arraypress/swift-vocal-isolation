import sys, warnings, yaml, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import stft
cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
m = MelBandRoformer(**mcfg).eval()
spec = stft(torch.randn(1, 2, cfg["audio"]["chunk_size"]) * 0.05, m)
print("spec", tuple(spec.shape))
print("freq_indices", tuple(m.freq_indices.shape), "dtype", m.freq_indices.dtype)
print("num_bands_per_freq", tuple(m.num_bands_per_freq.shape))
print("audio_channels", m.audio_channels, "| layers", len(m.layers), "| block len", len(m.layers[0]))
x = spec[:, m.freq_indices]
print("after gather", tuple(x.shape))
from einops import rearrange
x = rearrange(x, "b f t c -> b t (f c)")
x = m.band_split(x)
print("after band_split", tuple(x.shape))
print("mask_estimators", len(m.mask_estimators))
