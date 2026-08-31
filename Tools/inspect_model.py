import sys, warnings, yaml, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "_ref/mss")
from models.bs_roformer.mel_band_roformer import MelBandRoformer

cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"])
mcfg["flash_attn"] = False                     # export-safe attention path
print("model config:", {k: mcfg[k] for k in ("dim","depth","num_bands","heads","dim_head","stft_n_fft","stft_hop_length","num_stems")})

model = MelBandRoformer(**mcfg)
n = sum(p.numel() for p in model.parameters())
print(f"params: {n/1e6:.1f} M  -> fp32 {n*4/1e6:.0f} MB, fp16 {n*2/1e6:.0f} MB")

ck = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
sd = { (k[6:] if k.startswith("model.") else k): v for k, v in sd.items() }
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:5]:    print("  missing e.g.:", missing[:5])
if unexpected[:5]: print("  unexpected e.g.:", unexpected[:5])

model.eval()
audio = cfg["audio"]
chunk = audio["chunk_size"]
print(f"chunk_size: {chunk} samples = {chunk/audio['sample_rate']:.2f} s")
with torch.no_grad():
    out = model(torch.randn(1, 2, chunk) * 0.1)
print("forward OK: in [1,2,%d] -> out %s" % (chunk, tuple(out.shape)))
