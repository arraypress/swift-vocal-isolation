import sys, warnings, yaml, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
import coremltools as ct
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import RoformerCore, stft, staticize_rotary
cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
m = MelBandRoformer(**mcfg)
sd = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
m.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()})
m.eval()
spec = stft(torch.randn(1, 2, cfg["audio"]["chunk_size"]) * 0.05, m)
core = RoformerCore(m, frames=spec.shape[2], num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
staticize_rotary(m, lambda: core(spec))
print("exporting...", flush=True)
with torch.no_grad():
    ep = torch.export.export(core, (spec,))
ep = ep.run_decompositions({})
print("torch.export + decompositions OK", flush=True)
from roformer_core import strip_aliases
print("stripped aliases:", strip_aliases(ep), flush=True)
ml = ct.convert(ep, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS26)
ml.save("mel_band_roformer.mlpackage")
print("SAVED", flush=True)
