#!/usr/bin/env python3
"""Find where NaN enters the brain during the closed loop."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import DT, Brain  # noqa: E402
from bridge import BridgeParams, odour_to_orn_rates  # noqa: E402

N = json.loads((REPO / "neurons.v783.json").read_text())
orn_l = N["ORN"]["left"]["root_ids"]
orn_r = N["ORN"]["right"]["root_ids"]
dn_ids = [N["DN"]["DNa02"]["left"]["root_ids"][0],
          N["DN"]["DNa02"]["right"]["root_ids"][0]]

p = BridgeParams()
brain = Brain(batch=1, stim_ids=orn_l + orn_r, device="cuda")
li = torch.as_tensor(brain.idx(orn_l), device="cuda")
ri = torch.as_tensor(brain.idx(orn_r), device="cuda")
dn_cols = torch.as_tensor(brain.idx(dn_ids), device="cuda")


def state_health(tag):
    st = brain.state
    bad = {}
    for nm in ("conductance", "delay_buffer", "spikes", "v", "refrac"):
        t = getattr(st, nm)
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        mx = float(t.abs().max())
        if nan or inf:
            bad[nm] = (nan, inf)
        print(f"    {tag:12s} {nm:13s} nan={nan:7d} inf={inf:7d} absmax={mx:.3e}")
    return bad


# Realistic spawn-time odour: source at (22,11,1.5), fly at origin -> d~24.6mm
fake_obs = np.array([[0.00165, 0.00160, 0.00164, 0.00159], [0, 0, 0, 0]])
l_hz, r_hz, contrast, mean_i = odour_to_orn_rates(fake_obs, p)
print(f"odour -> ORN rates: L={l_hz:.2f} Hz  R={r_hz:.2f} Hz  "
      f"contrast={contrast:+.5f}  mean_i={mean_i:.6f}")

rates = brain.zero_rates()
rates[:, li] = l_hz
rates[:, ri] = r_hz
print(f"rates finite: {bool(torch.isfinite(rates).all())}  "
      f"max={float(rates.max()):.2f}  nonzero={int((rates > 0).sum())}")

brain.reset(seed=0)
print("\nrunning, checking state every 500 steps:")
with torch.no_grad():
    for i in range(1, 4001):
        s = brain.step(rates)
        if i % 500 == 0:
            nan_s = int(torch.isnan(s).sum())
            print(f"  step {i:5d}  spikes nan={nan_s}  "
                  f"n_spiking={int((s > 0).sum())}")
            bad = state_health(f"@{i}")
            if bad:
                print(f"  >>> FIRST NaN/Inf IN STATE at step {i}: {bad}")
                break

print("\nnarrowing: step-by-step v growth from a fresh reset")
brain.reset(seed=0)
with torch.no_grad():
    for i in range(1, 4001):
        brain.step(rates)
        v = brain.state.v
        c = brain.state.conductance
        if i % 250 == 0 or not bool(torch.isfinite(v).all()):
            print(f"  step {i:5d}  |v|max={float(v.abs().max()):.4e}  "
                  f"|g|max={float(c.abs().max()):.4e}  "
                  f"v finite={bool(torch.isfinite(v).all())}")
        if not bool(torch.isfinite(v).all()):
            nz = torch.nonzero(~torch.isfinite(v))
            print(f"  >>> v went non-finite at step {i}, "
                  f"{nz.shape[0]} entries, first idx {nz[0].tolist()}")
            break

print("\nfor comparison: the M1 sugar protocol (21 GRNs @ 200 Hz) over 4000 steps")
brain2 = Brain(batch=1, stim_ids=orn_l + orn_r, device="cuda", quiet=True)
r2 = brain2.zero_rates()
r2[:, li] = 200.0
brain2.reset(seed=0)
with torch.no_grad():
    for i in range(1, 4001):
        brain2.step(r2)
print(f"  after 4000 steps at 200 Hz on left ORNs: "
      f"v finite={bool(torch.isfinite(brain2.state.v).all())}  "
      f"|v|max={float(brain2.state.v.abs().max()):.4e}")
