# Copyright (c) 2026 David Sherlock
# Use of this source code is governed by an MIT license; see LICENSE.
"""Export Mel-Band Roformer's core (spectrogram -> mask) as a Core ML .mlpackage.

    .venv/bin/python export_coreml.py [out.mlpackage]

The STFT and ISTFT stay on the host in vDSP, as they already do for MDX23C, so the
graph is just the network. `flash_attn` is forced off: the flash path wraps SDPA in
`torch.backends.cuda.sdp_kernel`, a CUDA context manager that tracing cannot follow.
The non-flash path is plain einsum + softmax and is mathematically identical.
"""
import sys, warnings, yaml, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "_ref/mss")
import coremltools as ct
from models.bs_roformer.mel_band_roformer import MelBandRoformer
from roformer_core import RoformerCore, stft

OUT = sys.argv[1] if len(sys.argv) > 1 else "mel_band_roformer.mlpackage"

cfg = yaml.load(open("config_kj.yaml"), Loader=yaml.FullLoader)
mcfg = dict(cfg["model"]); mcfg["flash_attn"] = False
model = MelBandRoformer(**mcfg)
sd = torch.load("MelBandRoformer.ckpt", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
model.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()})
model.eval()

chunk = cfg["audio"]["chunk_size"]
with torch.no_grad():
    example = stft(torch.randn(1, 2, chunk) * 0.05, model)
print("input shape:", tuple(example.shape), flush=True)

core = RoformerCore(model, frames=example.shape[2],
                    num_bands=mcfg["num_bands"], dim=mcfg["dim"]).eval()
with torch.no_grad():
    traced = torch.jit.trace(core, example, check_trace=False)
print("traced", flush=True)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="spec", shape=example.shape, dtype=None)],
    outputs=[ct.TensorType(name="mask", dtype=None)],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS26,
)
mlmodel.short_description = ("Mel-Band Roformer vocal separation core (Kimberley Jensen "
                             "weights, MIT). Spectrogram in, complex mask out; STFT/ISTFT "
                             "on the host.")
mlmodel.save(OUT)
print("saved", OUT, flush=True)
