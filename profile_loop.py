#!/usr/bin/env python3
"""Where does the closed loop actually spend its time?

Measured serially on the one GPU, with warmup and explicit synchronisation.
This is the ground truth any proposed optimisation has to beat.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import DT, Brain  # noqa: E402

OUT = REPO / "out"
OUT.mkdir(exist_ok=True)
N = json.loads((REPO / "neurons.v783.json").read_text())


def bench(fn, n=200, warmup=30):
    """Return mean ms per call, synchronising properly."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


results = {}

rule("setup")
orn = N["ORN"]["left"]["root_ids"] + N["ORN"]["right"]["root_ids"]
brain = Brain(batch=1, stim_ids=orn, device="cuda")
dn_ids = [N["DN"]["DNa02"]["left"]["root_ids"][0],
          N["DN"]["DNa02"]["right"]["root_ids"][0],
          N["DN"]["DNa01"]["left"]["root_ids"][0],
          N["DN"]["DNa01"]["right"]["root_ids"][0]]
dn_cols = torch.as_tensor(brain.idx(dn_ids), device="cuda")
rates = brain.rates_for(orn, 100.0)
brain.reset(seed=0)
NN = brain.n_neurons
print(f"  neurons={NN}  weights={type(brain.weights).__name__} "
      f"layout={brain.weights.layout}  nnz={brain.weights._nnz() if hasattr(brain.weights,'_nnz') else '?'}")
print(f"  VRAM={brain.vram_gb():.2f} GB")

# ------------------------------------------------------------------ brain
rule("1. brain.step() as it stands")
ms = bench(lambda: brain.step(rates))
results["brain_step_ms"] = ms
print(f"  brain.step()                     {ms:8.3f} ms   "
      f"-> {DT/ms*1000:.0f} sim-ms per wall-s")

rule("2. brain step, component by component (same shapes, isolated)")
st = brain.state
m = brain.model
comp = {}

comp["bernoulli(rates)*scale"] = bench(
    lambda: torch.bernoulli(rates * (DT / 1000.0)) * 250.0)

sp = st.spikes
W = brain.weights
comp["matmul(spikes, W.T)  [sparse CSR]"] = bench(
    lambda: torch.matmul(sp, W.transpose(0, 1)))

# The transpose itself -- is it free, or materialising every call?
comp["  W.transpose(0,1) alone"] = bench(lambda: W.transpose(0, 1), n=500)

db = st.delay_buffer
comp["torch.roll(delay_buffer)"] = bench(lambda: torch.roll(db, shifts=-1, dims=1))
print(f"  (delay_buffer is {tuple(db.shape)} = "
      f"{db.numel() * 4 / 1e6:.1f} MB)")

cond = st.conductance
rf = st.refrac
comp["conductance arithmetic"] = bench(
    lambda: cond * (1 - DT / 5.0) + db[:, 0, :] * rf)

v = st.v
comp["LIF v update + threshold"] = bench(
    lambda: (v + 0.1 + (DT / 20.0) * (cond - (v + 52.0)) > -45.0).float())

comp["torch.where(refrac)"] = bench(
    lambda: torch.where(sp > 0, torch.zeros_like(rf), rf + 1))

for k, vv in sorted(comp.items(), key=lambda x: -x[1]):
    share = vv / ms * 100
    print(f"  {k:42s} {vv:8.3f} ms  ({share:5.1f}% of a step)")
results["brain_components_ms"] = comp

# ---------------------------------------------------------------- recording
rule("3. the per-step recording overhead")
rec = {}
rec["index_select(1, dn_cols)"] = bench(lambda: sp.index_select(1, dn_cols))
rec["advanced index sp[:, dn_cols]"] = bench(lambda: sp[:, dn_cols])
raster = torch.zeros(25000, 4, device="cuda")
d = sp.index_select(1, dn_cols)
rec["raster[k] = d[0]"] = bench(lambda: raster.__setitem__(7, d[0]))
counts = torch.zeros(1, 4, device="cuda")
rec["counts += d"] = bench(lambda: counts.add_(d))
for k, vv in sorted(rec.items(), key=lambda x: -x[1]):
    print(f"  {k:42s} {vv:8.3f} ms  ({vv/ms*100:5.1f}% of a brain step)")
results["recording_ms"] = rec

per_step_rec = (rec["index_select(1, dn_cols)"] + rec["raster[k] = d[0]"]
                + rec["counts += d"])
print(f"\n  total per-step recording: {per_step_rec:.3f} ms "
      f"= {per_step_rec/(ms+per_step_rec)*100:.1f}% of the inner loop")
results["recording_total_ms"] = per_step_rec

# --------------------------------------------------------------------- sync
rule("4. host synchronisation cost")
syn = {}
syn[".cpu() on (1,4)"] = bench(lambda: counts.cpu(), n=100)
syn[".item() on a scalar"] = bench(lambda: counts.sum().item(), n=100)
syn["torch.cuda.synchronize()"] = bench(lambda: torch.cuda.synchronize(), n=100)
for k, vv in sorted(syn.items(), key=lambda x: -x[1]):
    print(f"  {k:42s} {vv:8.3f} ms")
results["sync_ms"] = syn

# ------------------------------------------------------------------ physics
rule("5. MuJoCo / FlyGym cost")
try:
    from m4_loop import build_sim
    sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
    obs, _ = sim.reset(seed=0)
    act = np.array([1.0, 1.0], dtype=np.float32)

    for _ in range(50):
        sim.step(act)
    t0 = time.perf_counter()
    for _ in range(300):
        sim.step(act)
    phys_ms = (time.perf_counter() - t0) / 300 * 1000
    results["mujoco_step_ms"] = phys_ms
    print(f"  sim.step(action)                 {phys_ms:8.3f} ms   "
          f"-> {0.1/phys_ms*1000:.0f} sim-ms per wall-s")

    t0 = time.perf_counter()
    for _ in range(30):
        sim.render()
    rend_ms = (time.perf_counter() - t0) / 30 * 1000
    results["render_ms"] = rend_ms
    print(f"  sim.render()                     {rend_ms:8.3f} ms  "
          f"(called once per exchange)")
    sim.close()
except Exception as e:
    print(f"  physics bench failed: {type(e).__name__}: {e}")
    phys_ms, rend_ms = float("nan"), float("nan")

# ------------------------------------------------------------------ budget
rule("6. BUDGET for one 5 ms exchange (50 brain + 50 physics steps)")
b_tot = (ms + per_step_rec) * 50
p_tot = phys_ms * 50
s_tot = syn[".cpu() on (1,4)"]
r_tot = rend_ms
total = b_tot + p_tot + s_tot + r_tot
rows = [
    ("brain steps (50x)", (ms) * 50),
    ("  of which torch.roll", comp["torch.roll(delay_buffer)"] * 50),
    ("  of which sparse matmul", comp["matmul(spikes, W.T)  [sparse CSR]"] * 50),
    ("per-step recording (50x)", per_step_rec * 50),
    ("physics steps (50x)", p_tot),
    ("render (1x)", r_tot),
    ("host sync (1x)", s_tot),
]
for k, vv in rows:
    print(f"  {k:36s} {vv:9.2f} ms  ({vv/total*100:5.1f}%)")
print(f"  {'TOTAL':36s} {total:9.2f} ms per 5 ms of simulated time")
print(f"  => realtime factor {5.0/total:.4f}x "
      f"(measured end-to-end previously: 0.013x)")
results["budget_ms"] = dict(rows)
results["exchange_total_ms"] = total
results["realtime_factor"] = 5.0 / total

(OUT / "profile.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT/'profile.json'}")
