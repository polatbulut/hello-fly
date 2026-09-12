"""Steppable wrapper around the FlyWire v783 LIF whole-brain model.

The upstream runner (eonsystemspbc/fly-brain, code/run_pytorch.py) is written as a
benchmark: it builds a model and then runs a fixed-length `for` loop internally.
For a closed loop we need to own that loop, so this module reuses upstream's
model classes verbatim -- same LIF equations, same MODEL_PARAMS, same weights --
and exposes them as an explicit `step()`.

Nothing about the neuroscience is changed here. `TorchModel.forward()` already IS
a pure one-timestep function: all state goes in, all state comes out, no hidden
mutation. This class just holds that state and adds the things a closed loop
needs: root-ID -> tensor-index lookup, per-neuron rate injection, smoothed rate
readout, and output silencing for ablations.

Timestep is DT = 0.1 ms, inherited from upstream (matches Brian2 defaultclock.dt).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pyarrow  # noqa: F401  -- upstream requires this imported before torch
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
FLYBRAIN = REPO / "external" / "fly-brain"
if str(FLYBRAIN / "code") not in sys.path:
    sys.path.insert(0, str(FLYBRAIN / "code"))

from run_pytorch import (  # noqa: E402
    DT,
    MODEL_PARAMS,
    TorchModel,
    get_hash_tables,
    get_weights,
)

PATH_COMP = FLYBRAIN / "data" / "2025_Completeness_783.csv"
PATH_CONN = FLYBRAIN / "data" / "2025_Connectivity_783.parquet"
PATH_WT = FLYBRAIN / "data"

__all__ = ["Brain", "RateReadout", "DT", "MODEL_PARAMS"]


@dataclass
class _State:
    conductance: torch.Tensor
    delay_buffer: torch.Tensor
    spikes: torch.Tensor
    v: torch.Tensor
    refrac: torch.Tensor


class Brain:
    """The v783 LIF brain, advanced one 0.1 ms timestep at a time.

    Parameters
    ----------
    batch : int
        Independent trials run in parallel on the GPU. The closed loop uses 1;
        benchmarking and statistics use more.
    stim_ids : sequence[int] | None
        FlyWire root IDs to treat as externally driven. Upstream zeroes the
        refractory period for these so they can follow a high Poisson rate --
        we preserve that, since it is part of the published protocol.
    device : str
    """

    def __init__(self, batch=1, stim_ids=None, device="cuda", quiet=False):
        self.device = device
        self.batch = batch
        self._log = (lambda *a: None) if quiet else print

        self._log("loading ID map...")
        self.flyid2i, self.i2flyid = get_hash_tables(str(PATH_COMP))
        self.n_neurons = len(self.flyid2i)

        stim_ids = list(stim_ids or [])
        missing = [r for r in stim_ids if r not in self.flyid2i]
        if missing:
            raise KeyError(
                f"{len(missing)} stim root ID(s) absent from v783: {missing[:5]}"
            )
        self.stim_idx = [self.flyid2i[r] for r in stim_ids]

        self._log(f"loading weights ({self.n_neurons} neurons)...")
        weights = get_weights(str(PATH_CONN), str(PATH_COMP), str(PATH_WT), csr=True)
        self.weights = weights.to(device=device)

        self.model = TorchModel(
            batch, self.n_neurons, DT, MODEL_PARAMS, self.weights,
            exc_indices=self.stim_idx or None, device=device,
        )

        # Output-silencing mask, multiplied into `spikes` after every step.
        # 1.0 = normal, 0.0 = silenced. This suppresses the neuron's effect on
        # every downstream partner (spikes are what the recurrent matmul reads)
        # while leaving its own membrane dynamics running -- the model analogue
        # of an optogenetic output block, and the ablation used in M5.
        #
        # NOTE this differs from upstream's GeNN backend, which implements
        # silencing by deleting connectome edges before building. Deleting edges
        # also removes the neuron's *inputs*; masking spikes does not. For an
        # ablation asking "does this neuron's output drive the behaviour?",
        # masking is the more direct question.
        self._mask = None
        self.state = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def idx(self, root_ids):
        """FlyWire root IDs -> tensor column indices. Raises on unknown IDs."""
        root_ids = np.atleast_1d(root_ids)
        miss = [int(r) for r in root_ids if int(r) not in self.flyid2i]
        if miss:
            raise KeyError(f"root ID(s) not in v783: {miss[:5]}")
        return np.array([self.flyid2i[int(r)] for r in root_ids], dtype=np.int64)

    def silence(self, root_ids):
        """Zero the spike output of these neurons for all subsequent steps."""
        if root_ids is None or len(root_ids) == 0:
            self._mask = None
            return np.array([], dtype=np.int64)
        i = self.idx(root_ids)
        m = torch.ones(self.n_neurons, device=self.device)
        m[torch.as_tensor(i, device=self.device)] = 0.0
        self._mask = m
        return i

    def unsilence(self):
        self._mask = None

    def reset(self, seed=None):
        """Re-initialise membrane state. Call between trials."""
        if seed is not None:
            torch.manual_seed(seed)
        c, d, s, v, r = self.model.state_init()
        self.state = _State(c, d, s, v, r)
        self.t_ms = 0.0
        return self

    def zero_rates(self):
        return torch.zeros(self.batch, self.n_neurons, device=self.device)

    # ------------------------------------------------------------------- run
    @torch.no_grad()
    def step(self, rates):
        """Advance exactly one DT (0.1 ms). Returns the spike tensor (batch, N)."""
        st = self.state
        c, d, s, v, r = self.model(
            rates, st.conductance, st.delay_buffer, st.spikes, st.v, st.refrac
        )
        if self._mask is not None:
            s = s * self._mask
        self.state = _State(c, d, s, v, r)
        self.t_ms += DT
        return s

    @torch.no_grad()
    def run(self, rates, duration_ms, record_idx=None, record_all=False):
        """Advance `duration_ms`, optionally recording spikes.

        record_idx  : record only these columns, as a dense (steps, batch, k) array
        record_all  : record every spike sparsely, as (t_ms, trial, neuron_index)

        Recording everything for long runs is memory-hungry -- 1 s at 0.1 ms is
        10,000 steps -- so the closed loop records only the DN columns.
        """
        n_steps = int(round(duration_ms / DT))
        dense, ev_t, ev_b, ev_n = [], [], [], []
        cols = None
        if record_idx is not None:
            cols = torch.as_tensor(np.asarray(record_idx), device=self.device)

        for k in range(n_steps):
            s = self.step(rates)
            if cols is not None:
                dense.append(s.index_select(1, cols).to("cpu", non_blocking=True))
            if record_all:
                hit = s > 0
                if hit.any():
                    b, n = hit.nonzero(as_tuple=True)
                    ev_b.append(b.cpu())
                    ev_n.append(n.cpu())
                    ev_t.append(torch.full((len(b),), k, dtype=torch.long))

        out = {}
        if cols is not None:
            out["dense"] = torch.stack(dense).numpy()          # (steps, batch, k)
        if record_all:
            if ev_t:
                out["spikes"] = pd.DataFrame({
                    "t_ms": torch.cat(ev_t).numpy() * DT,
                    "trial": torch.cat(ev_b).numpy(),
                    "neuron_index": torch.cat(ev_n).numpy(),
                })
            else:
                out["spikes"] = pd.DataFrame(
                    columns=["t_ms", "trial", "neuron_index"])
        return out

    def rates_for(self, root_ids, hz, out=None):
        """Build a rate tensor with `hz` on the given neurons, 0 elsewhere."""
        r = self.zero_rates() if out is None else out
        r[:, torch.as_tensor(self.idx(root_ids), device=self.device)] = hz
        return r

    def vram_gb(self):
        if self.device != "cuda":
            return 0.0
        free, total = torch.cuda.mem_get_info(self.device)
        return (total - free) / 1024 ** 3


class RateReadout:
    """Exponentially smoothed firing rate, in Hz, from 0/1 spike samples.

    A single 0.1 ms timestep carries essentially no rate information (a spike is
    either there or not, so the instantaneous estimate is 0 Hz or 10,000 Hz).
    The closed loop needs a usable continuous signal, so we low-pass with
    time constant `tau_ms`.

    tau_ms is a BRIDGE PARAMETER WITH NO BIOLOGICAL SOURCE -- it sets how much
    the steering signal lags and smooths. Flagged in docs/bridge-parameters.md.
    """

    def __init__(self, n, tau_ms=50.0, device="cuda", batch=1):
        self.alpha = DT / tau_ms
        self.tau_ms = tau_ms
        self.rate = torch.zeros(batch, n, device=device)

    def update(self, spikes_subset):
        inst = spikes_subset / (DT / 1000.0)        # spikes/step -> Hz
        self.rate += self.alpha * (inst - self.rate)
        return self.rate

    def reset(self):
        self.rate.zero_()
        return self
