#!/usr/bin/env python3
"""THE GATE: does FastBrain still reproduce M1, and how much faster is it?

The optimisations were each verified bit-exact in isolation. That is necessary
but not sufficient -- what matters is the end-to-end validated result. M1's
anchor is upstream's own v783 manifest: 16,353-17,429 spikes and 380-416 active
neurons for 21 sugar GRNs at 200 Hz over 1 s.

Additionally: with a fixed seed, FastBrain and Brain must produce the SAME spike
train, not merely a statistically similar one.
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import DT, Brain  # noqa: E402
from fast import FastBrain  # noqa: E402

sys.path.insert(0, str(REPO / "external/fly-brain/code"))
from benchmark import EXPERIMENTS  # noqa: E402

ANCHOR_SPIKES = (16353, 17429)
ANCHOR_ACTIVE = (380, 416)


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


exp = EXPERIMENTS["sugar"]
grns, hz = exp["neu_exc"], exp["stim_rate"]

rule("A. step-for-step identity at a fixed seed (500 steps)")
b_ref = Brain(batch=1, stim_ids=grns, device="cuda", quiet=True)
b_fast = FastBrain(batch=1, stim_ids=grns, device="cuda", quiet=True)

r_ref = b_ref.rates_for(grns, hz)
r_fast = b_fast.rates_for(grns, hz)

mismatch, first_bad = 0, None
with torch.no_grad():
    torch.manual_seed(1234)
    b_ref.reset()
    s_ref_all = [b_ref.step(r_ref).clone() for _ in range(500)]
    torch.manual_seed(1234)
    b_fast.reset()
    s_fast_all = [b_fast.step(r_fast).clone() for _ in range(500)]

for i, (a, b) in enumerate(zip(s_ref_all, s_fast_all)):
    if not torch.equal(a, b):
        mismatch += 1
        if first_bad is None:
            first_bad = i
print(f"  identical steps : {500 - mismatch}/500")
if mismatch:
    print(f"  first divergence at step {first_bad}")
    print("  => NOT bit-exact end to end. Do not adopt.")
else:
    print("  => BIT-EXACT end to end at a fixed seed")

rule("B. M1 anchor, FastBrain, 1 s of sugar GRNs at 200 Hz")
for name, B in (("Brain (reference)", Brain), ("FastBrain", FastBrain)):
    br = B(batch=1, stim_ids=grns, device="cuda", quiet=True)
    rates = br.rates_for(grns, hz)
    br.reset(seed=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = br.run(rates, duration_ms=1000.0, record_all=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    sp = res["spikes"]
    n, a = len(sp), sp["neuron_index"].nunique()
    ok_s = ANCHOR_SPIKES[0] <= n <= ANCHOR_SPIKES[1]
    ok_a = ANCHOR_ACTIVE[0] <= a <= ANCHOR_ACTIVE[1]
    print(f"  {name:18s} spikes={n:6d} {'ok' if ok_s else 'OUT OF BAND'}   "
          f"active={a:4d} {'ok' if ok_a else 'OUT OF BAND'}   "
          f"wall={wall:6.2f} s  ({1000/wall:6.1f} sim-ms/wall-s)")
    del br
    torch.cuda.empty_cache()

rule("C. raw step throughput")
out = {}
for name, B in (("Brain", Brain), ("FastBrain", FastBrain)):
    br = B(batch=1, stim_ids=grns, device="cuda", quiet=True)
    rates = br.rates_for(grns, hz)
    br.reset(seed=0)
    for _ in range(50):
        br.step(rates)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(400):
        br.step(rates)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 400 * 1000
    out[name] = ms
    print(f"  {name:12s} {ms:7.4f} ms/step   "
          f"{DT/ms*1000:7.1f} sim-ms per wall-second "
          f"({DT/ms:.4f}x realtime)")
    del br
    torch.cuda.empty_cache()

sp_up = out["Brain"] / out["FastBrain"]
print(f"\n  brain speedup: {sp_up:.2f}x  "
      f"({out['Brain']*50:.1f} -> {out['FastBrain']*50:.1f} ms per 50-step exchange)")

rule("VERDICT")
print(f"  bit-exact vs reference : {'YES' if mismatch == 0 else 'NO'}")
print(f"  brain speedup          : {sp_up:.2f}x")
print(f"  saving per exchange    : {(out['Brain']-out['FastBrain'])*50:.1f} ms")
(REPO / "out" / "fast_verify.json").write_text(json.dumps({
    "bit_exact": mismatch == 0, "ms_per_step": out, "speedup": sp_up,
}, indent=2))
