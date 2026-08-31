import sys, warnings, yaml, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "_ref/mss")
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import RoformerCore, stft
cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
m = MelBandRoformer(**mcfg).eval()
spec = stft(torch.randn(1, 2, cfg["audio"]["chunk_size"]) * 0.05, m)
core = RoformerCore(m, frames=spec.shape[2], num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
with torch.no_grad():
    tr = torch.jit.trace(core, spec, check_trace=False)
g = tr.inlined_graph
nodes = [n for n in g.nodes()]
hits = [n for n in nodes if n.kind() == "aten::Int"]
print("aten::Int nodes:", len(hits))
for n in hits[:6]:
    inp = list(n.inputs())[0]
    print("---")
    print("  node:", str(n).strip()[:160])
    print("  input type:", inp.type())
    prod = inp.node()
    print("  produced by:", str(prod).strip()[:200])
    print("  scope:", prod.scopeName()[:200])
