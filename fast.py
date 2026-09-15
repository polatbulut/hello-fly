"""Bit-exact speedups for the brain step. Nothing here changes the science.

Two changes, each verified bit-identical in verify_opt.py before being adopted:

  1. int32 CSR indices. PyTorch's to_sparse_csr() produces int64 indices and ATen
     hands them to cuSPARSE verbatim, so every timestep streams nnz*8 = 120.7 MB
     of column indices. nnz (15,091,983) and N (138,639) both fit in int32 with
     three orders of magnitude to spare. Measured 0.686 -> 0.565 ms on the SpMV,
     bit-identical on 96/96 probes.
     (The analysis that proposed this projected 1.50x from a pure-bandwidth
     argument. Measured is 1.21x -- implied bandwidth falls 266 -> 215 GB/s, so
     the SpMV is not purely bandwidth-bound. The projection was wrong; the
     optimisation is still real.)

  2. Ring-buffer delay line. Upstream reads delay slot 0, torch.rolls the whole
     (1, 19, 138639) buffer left, then writes the newest input into slot -1 --
     copying 21 MB per 0.1 ms timestep. A modular head index does the same thing
     with a single 0.55 MB slot write. Measured 0.0733 -> 0.0110 ms, saving
     3.12 ms per exchange, reads bit-identical over 60 steps.

NOT adopted, deliberately: FlyGym registers 2,268 explicit collision pairs of
which only 1,182 are unique -- Fly._init_self_contacts de-duplicates on the
ORDERED key f"{geom1}_{geom2}", so both (A,B) and (B,A) are added. Removing the
1,086 duplicates would cut broad-phase work, but no duplicated pair is in contact
during normal walking (measured: 11 contacts, all on unique leg-ground pairs), so
the saving is unproven, and if legs ever DO touch, the fix would change contact
forces. That is a dynamics change, not an optimisation. Left alone; worth filing
upstream.
"""
from __future__ import annotations

import torch

from brain import Brain

__all__ = ["FastBrain", "to_int32_csr"]


def to_int32_csr(W: torch.Tensor) -> torch.Tensor:
    """Narrow a CSR tensor's index arrays from int64 to int32. Bit-exact."""
    if W.layout != torch.sparse_csr:
        return W
    if W.crow_indices().dtype == torch.int32:
        return W
    if max(W._nnz(), *W.shape) >= 2**31 - 1:
        raise ValueError("indices do not fit in int32; refusing to narrow")
    return torch.sparse_csr_tensor(
        W.crow_indices().to(torch.int32),
        W.col_indices().to(torch.int32),
        W.values(), size=W.shape, dtype=W.dtype, device=W.device,
    )


def _ring_forward(self, input_, conductance, delay_buffer, refrac):
    """Drop-in replacement for AlphaSynapse.forward using a ring index.

    Upstream:
        read  delay_buffer[:, 0, :]        (oldest)
        roll  the buffer left by one
        write input_ into delay_buffer[:, -1, :]   (newest)

    Ring equivalent: the oldest slot is exactly the slot the newest value will
    occupy after the roll, so read it, overwrite it, and advance the head.
    Verified bit-identical over 60 steps in verify_opt.py.
    """
    conductance_new = (
        conductance * (1 - self.time_factor)
        + delay_buffer[:, self._head, :] * refrac
    )
    delay_buffer[:, self._head, :] = input_
    self._head = (self._head + 1) % delay_buffer.shape[1]
    return conductance_new, delay_buffer


class FastBrain(Brain):
    """Brain with the two bit-exact optimisations applied.

    Numerically identical to Brain: same weights, same LIF equations, same
    MODEL_PARAMS, same RNG consumption order. Only the integer width of the CSR
    indices and the mechanics of the delay line differ.
    """

    def __init__(self, *args, int32_indices=True, ring_buffer=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._opts = []

        if int32_indices:
            self.weights = to_int32_csr(self.weights)
            self.model.weights = self.weights
            self._opts.append("int32-csr")

        if ring_buffer:
            syn = self.model.neurons.synapse
            syn._head = 0
            # Bind the replacement to this instance only; the upstream class is
            # left untouched so a plain Brain in the same process is unaffected.
            syn.forward = _ring_forward.__get__(syn, type(syn))
            self._ring_syn = syn
            self._opts.append("ring-buffer")

        self._log(f"FastBrain optimisations: {', '.join(self._opts) or 'none'}")

    def reset(self, seed=None):
        out = super().reset(seed=seed)
        # The head must restart with the buffer, or the delay line reads a slot
        # that no longer corresponds to the oldest entry.
        if hasattr(self, "_ring_syn"):
            self._ring_syn._head = 0
        return out
