# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "coremltools", "ml_collections", "pyyaml", "soundfile"]
# ///
"""MDX23C (TFC-TDF-UNet v3, Kim & Lee; implementation ZFTurbo, MIT; weights
MDX23C-8KFFT-InstVoc_HQ by UVR, MIT) → stems-mdx23c-instvoc-float32.aimodel.

    uv run Tools/export_mdx.py --checkpoint MDX23C-8KFFT-InstVoc_HQ.ckpt --config model_2_stem_full_band_8k.yaml
        [--coreml mdx_net.mlpackage] [--out Tools/exports] [--install]

The asset holds the network between the spectrogram and the stem spectrograms, the same
cut the Core ML `mdx_net.mlpackage` makes, so `VocalIsolator`'s STFT, chunking and iSTFT
stay as they are: input `spec` [1, 4, 4096, 256] (L re, L im, R re, R im; bins 0..4095 of
n_fft 8192, hop 1024, 261,120 samples) → output `stems` [1, 8, 4096, 256] (vocals then
instrumental, 4 channels each). Asserted equal to upstream's `model(chunk)` before export.
Given `--coreml`, the old Core ML package is run on the same input to prove it is this
network — the provenance of the package was never written down.
"""
import argparse, shutil, sys, time
from pathlib import Path
import numpy as np, torch, yaml
from ml_collections import ConfigDict

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True); ap.add_argument("--config", required=True)
ap.add_argument("--coreml", help="the old mdx_net.mlpackage, to compare against")
ap.add_argument("--out", default=str(HERE / "exports")); ap.add_argument("--install", action="store_true")
ap.add_argument("--precision", choices=["float32", "float16"], default="float32",
                help="float16 halves the asset: weights and arithmetic in half, input and output still float32")
args = ap.parse_args()

sys.path.insert(0, str(HERE / "_ref/mss"))
from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net

config = ConfigDict(yaml.safe_load(open(args.config)))
model = TFC_TDF_net(config)
# weights_only: a checkpoint is a pickle, and this one came from the internet.
sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
for key in ("state", "state_dict"):
    if isinstance(sd, dict) and key in sd: sd = sd[key]
print("load:", model.load_state_dict(sd), flush=True)
model.eval()
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)


class Core(torch.nn.Module):
    """Upstream's forward from after the STFT to before the inverse STFT."""
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, spec):
        n = self.net
        mix = x = n.cac2cws(spec)
        first = x = n.first_conv(x)
        x = x.transpose(-1, -2)
        skips = []
        for block in n.encoder_blocks:
            x = block.tfc_tdf(x); skips.append(x); x = block.downscale(x)
        x = n.bottleneck_block(x)
        for block in n.decoder_blocks:
            x = block.upscale(x); x = torch.cat([x, skips.pop()], 1); x = block.tfc_tdf(x)
        x = x.transpose(-1, -2) * first
        x = n.final_conv(torch.cat([mix, x], 1))
        return n.cws2cac(x)


core = Core(model).eval()
L = int(config.audio.chunk_size)
torch.manual_seed(0)
chunk = torch.randn(1, 2, L) * 0.1
with torch.no_grad():
    spec = model.stft(chunk)
    stems = core(spec)
    b, c, f, t = stems.shape
    ours = model.stft.inverse(stems.reshape(b, model.num_target_instruments, -1, f, t))
    theirs = model(chunk)
print(f"spec {tuple(spec.shape)} → stems {tuple(stems.shape)}", flush=True)
print(f"core + iSTFT vs model(chunk): max |diff| {(ours - theirs).abs().max():.2e}", flush=True)
assert torch.equal(ours, theirs)

if args.coreml:
    import coremltools as ct
    old = ct.models.MLModel(args.coreml)
    spec_in = old.get_spec().description.input[0].name
    out = old.predict({spec_in: spec.numpy()})
    prediction = np.asarray(next(iter(out.values())), dtype=np.float32)
    ref = stems.numpy()
    snr = 10 * np.log10((ref ** 2).sum() / ((prediction - ref) ** 2).sum())
    print(f"old Core ML package vs this checkpoint: SNR {snr:.1f} dB, max |diff| {np.abs(prediction - ref).max():.3e}", flush=True)

import coreai_torch
from coreai.runtime import AIModelAssetMetadata


class Half(torch.nn.Module):
    """The core in half precision behind float32 input and output, so the host is unchanged."""
    def __init__(self, inner):
        super().__init__(); self.inner = inner.half()

    def forward(self, spec):
        return self.inner(spec.half()).float()


exported = Half(Core(model).eval()).eval() if args.precision == "float16" else core
# float16 is checked against the float32 reference on the GPU (Swift probe), not here:
# torch's half-precision convolutions on the CPU ran single-threaded for over 30 minutes.
t0 = time.time()
with torch.no_grad():
    ex = torch.export.export(exported, (spec,)).run_decompositions(coreai_torch.get_decomp_table())
print(f"exported in {time.time() - t0:.0f} s: {sum(1 for n in ex.graph.nodes if n.op == 'call_function')} ops", flush=True)
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
converter.add_exported_program(ex, input_names=["spec"], output_names=["stems"], entrypoint_name="main")
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Kim & Lee — TFC-TDF-UNet v3 (MDX23C); implementation ZFTurbo (MIT); weights UVR MDX23C-8KFFT-InstVoc_HQ (MIT); Core AI export by VocalIsolation"
meta.license = "MIT"
meta.model_description = ("MDX23C vocals/instrumental: spec [1,4,4096,256] (L re, L im, R re, R im; n_fft 8192, hop 1024, "
                          "261,120 samples) → stems [1,8,4096,256] (vocals ×4, instrumental ×4). STFT, chunking and iSTFT in the host.")
meta.creation_date = int(time.time())
dest = Path(args.out) / f"stems-mdx23c-instvoc-{args.precision}.aimodel"; dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists(): shutil.rmtree(dest)
program.save_asset(dest, meta)
print(f"saved {dest} ({sum(p.stat().st_size for p in dest.rglob('*')) / 1e6:.0f} MB) in {time.time() - t0:.0f} s", flush=True)
np.save(Path(args.out) / "mdx-probe-spec.npy", spec.numpy()); np.save(Path(args.out) / "mdx-probe-stems.npy", stems.numpy())  # float32 reference
if args.install:
    target = Path.home() / "Library/Application Support/stems" / dest.name
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(dest, target); print(f"installed {target}")
