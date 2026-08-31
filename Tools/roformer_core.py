# Copyright (c) 2026 David Sherlock
# Use of this source code is governed by an MIT license; see LICENSE.
"""Mel-Band Roformer with the STFT/ISTFT lifted out of the graph.

The Swift side already owns the transform (vDSP), so the exported model should be
spectrogram-in / mask-out. `RoformerCore` reproduces `MelBandRoformer.forward`
between those two points; `stft`/`apply_mask` mirror the halves that move to the host.

Everything runs in real arithmetic — the reference uses `view_as_complex` and a
complex `scatter_add_`, which Core ML has no type for. Complex addition is
componentwise, so scattering the real and imaginary planes together is equivalent.
"""
from __future__ import annotations

import torch
from torch import nn
from einops import rearrange, pack, unpack


class RoformerCore(nn.Module):
    """`[b, f*s, t, 2]` spectrogram -> `[b, f*s, t, 2]` complex mask.

    Every dimension is fixed at construction. The reference uses einops `pack`/`unpack`,
    which compute shapes from tensors at run time; tracing turns those into `aten::Int`
    on non-scalars, which Core ML cannot convert. Nothing here is dynamic, so the same
    reshapes are written out with Python ints instead.
    """

    def __init__(self, model: nn.Module, frames: int, num_bands: int,
                 dim: int, batch: int = 1) -> None:
        super().__init__()
        self.m = model
        self.B, self.T, self.NB, self.D = batch, frames, num_bands, dim
        self.FS = model.num_bands_per_freq.shape[0] * model.audio_channels
        self.NSEL = model.freq_indices.shape[0]
        # int32 indices: Core ML has no int64 tensor type.
        self.register_buffer("idx", model.freq_indices.to(torch.int32), persistent=False)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        m = self.m
        B, T, NB, D = self.B, self.T, self.NB, self.D

        x = torch.index_select(spec, 1, self.idx)          # [B, NSEL, T, 2]
        x = x.reshape(B, self.NSEL, T * 2).reshape(B, self.NSEL, T, 2)
        x = x.permute(0, 2, 1, 3).reshape(B, T, self.NSEL * 2)
        x = m.band_split(x)                                # [B, T, NB, D]

        for block in m.layers:
            time_transformer, freq_transformer = block[-2], block[-1]

            x = x.permute(0, 2, 1, 3).reshape(B * NB, T, D)
            x = time_transformer(x)
            x = x.reshape(B, NB, T, D).permute(0, 2, 1, 3)

            x = x.reshape(B * T, NB, D)
            x = freq_transformer(x)
            x = x.reshape(B, T, NB, D)

        mask = m.mask_estimators[0](x)                     # [B, T, NSEL*2]
        mask = mask.reshape(B, T, self.NSEL, 2).permute(0, 2, 1, 3)

        # Average the mask over bands sharing a frequency. Complex addition is
        # componentwise, so scattering both planes at once matches the reference's
        # complex `scatter_add_` exactly.
        idx = self.idx.to(torch.int64).view(1, -1, 1, 1).expand(B, self.NSEL, T, 2)
        summed = torch.zeros(B, self.FS, T, 2, dtype=mask.dtype, device=mask.device)
        summed = summed.scatter_add(1, idx, mask)

        denom = m.num_bands_per_freq.repeat_interleave(m.audio_channels)
        return summed / denom.view(1, -1, 1, 1).clamp(min=1e-8)


def stft(audio: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Host-side half: `[b, s, t]` audio -> `[b, f*s, frames, 2]`, as the model packs it."""
    b, s, _ = audio.shape
    flat = rearrange(audio, "b s t -> (b s) t")
    window = model.stft_window_fn(device=audio.device)
    spec = torch.stft(flat, **model.stft_kwargs, window=window, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = rearrange(spec, "(b s) f t c -> b (f s) t c", b=b, s=s)
    return spec


def apply_mask(spec: torch.Tensor, mask: torch.Tensor, model: nn.Module,
               length: int | None = None) -> torch.Tensor:
    """Host-side half: multiply, zero DC, inverse transform back to audio."""
    out = torch.view_as_complex(spec.contiguous()) * torch.view_as_complex(mask.contiguous())
    out = rearrange(out, "b (f s) t -> (b s) f t", s=model.audio_channels)
    if model.zero_dc:
        out[:, 0] = 0.0
    window = model.stft_window_fn(device=spec.device)
    audio = torch.istft(out, **model.stft_kwargs, window=window,
                        return_complex=False, length=length)
    return rearrange(audio, "(b s) t -> b s t", s=model.audio_channels)


def staticize_rotary(model: nn.Module, run_once) -> int:
    """Freeze RoPE at the sequence lengths this model actually uses.

    `rotate_queries_or_keys` slices its frequency table with `freqs[-seq_len:]`, where
    `seq_len` comes from the tensor's shape. Tracing turns that negative index into
    `aten::Int` on a non-scalar, which Core ML rejects — 280 such nodes in this graph.

    Every length here is fixed (801 time frames, 60 bands), so this runs the model once
    to learn which length each rotary module sees, evaluates the frequency table eagerly
    at that length, and swaps in a rotation that needs no shape arithmetic. The rotation
    itself is `apply_rotary_emb` with `start_index=0` and `scale=1`, unrolled.

    Returns the number of modules patched.
    """
    from rotary_embedding_torch import RotaryEmbedding
    from rotary_embedding_torch.rotary_embedding_torch import rotate_half

    lengths: dict[int, set[int]] = {}
    rotaries = [m for m in model.modules() if isinstance(m, RotaryEmbedding)]

    def recorder(mod, original):
        def wrapped(t, seq_dim=None, offset=0, freq_seq_len=None):
            dim = mod.default_seq_dim if seq_dim is None else seq_dim
            lengths.setdefault(id(mod), set()).add(int(t.shape[dim]))
            return original(t, seq_dim=seq_dim, offset=offset, freq_seq_len=freq_seq_len)
        return wrapped

    saved = {id(m): m.rotate_queries_or_keys for m in rotaries}
    for m in rotaries:
        m.rotate_queries_or_keys = recorder(m, saved[id(m)])
    with torch.no_grad():
        run_once()
    for m in rotaries:
        m.rotate_queries_or_keys = saved[id(m)]

    for m in rotaries:
        seen = lengths.get(id(m), set())
        if len(seen) != 1:
            raise RuntimeError(f"rotary module sees {sorted(seen)} lengths; expected exactly one")
        n = seen.pop()
        with torch.no_grad():
            pos = m.get_seq_pos(n, device="cpu", dtype=torch.float32)
            freqs = m(pos)                                   # [n, rot_dim]

        def static(t, seq_dim=None, offset=0, freq_seq_len=None, _f=freqs):
            f = _f.to(t)
            rot_dim = f.shape[-1]
            head, tail = t[..., :rot_dim], t[..., rot_dim:]
            head = head * f.cos() + rotate_half(head) * f.sin()
            return torch.cat((head, tail), dim=-1) if tail.shape[-1] else head

        m.rotate_queries_or_keys = static

    return len(rotaries)


def strip_aliases(ep):
    """Remove `aten.alias` nodes from an ExportedProgram.

    Decomposition leaves `alias` nodes behind and coremltools' EXIR frontend has no
    handler for them ("Unsupported fx node alias"). `alias` returns its input
    unchanged, so forwarding each node's uses to its argument is exact, not an
    approximation. Returns the number removed.
    """
    targets = {torch.ops.aten.alias.default, torch.ops.aten.alias_copy.default}
    graph = ep.graph_module.graph
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in targets:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            removed += 1
    graph.lint()
    ep.graph_module.recompile()
    return removed
