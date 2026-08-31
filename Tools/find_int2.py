import sys, warnings, yaml, torch, collections
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import RoformerCore, stft, staticize_rotary
cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
m = MelBandRoformer(**mcfg).eval()
spec = stft(torch.randn(1, 2, cfg["audio"]["chunk_size"]) * 0.05, m)
core = RoformerCore(m, frames=spec.shape[2], num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
staticize_rotary(m, lambda: core(spec))
with torch.no_grad():
    tr = torch.jit.trace(core, spec, check_trace=False)
hits = [n for n in tr.inlined_graph.nodes() if n.kind() == "aten::Int"]
kinds = collections.Counter(list(n.inputs())[0].node().kind() for n in hits)
print("producers:", dict(kinds))
scopes = collections.Counter(list(n.inputs())[0].node().scopeName().split("/")[-1] for n in hits)
for s, c in scopes.most_common(6): print(f"  {c:>4}  {s[:110]}")
print("--- sample ---")
for n in hits[:3]:
    p = list(n.inputs())[0].node()
    print(" ", str(p).strip()[:190])
