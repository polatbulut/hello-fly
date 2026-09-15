#!/usr/bin/env python3
"""Verify the proposed optimisations are BIT-EXACT before trusting any speedup.

The verifiers were explicit: measure, do not project. Three claims under test:
  1. int32 CSR indices          -- projected 0.687 -> 0.459 ms, claimed EXACT
  2. ring-buffer delay line     -- replaces torch.roll, claimed EXACT
  3. FlyGym self-contact pairs  -- claimed ~2x duplicated by an ordered-key bug

Correctness first, speed second. A faster wrong answer is a failure.
"""
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


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def bench(fn, n=200, warmup=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ===================================================================== 1
rule("1. int32 CSR indices -- bit-exact?")
import json
N = json.loads((REPO / "neurons.v783.json").read_text())
orn = N["ORN"]["left"]["root_ids"] + N["ORN"]["right"]["root_ids"]
brain = Brain(batch=1, stim_ids=orn, device="cuda")
W64 = brain.weights
print(f"  nnz={W64._nnz():,}  N={W64.shape[0]:,}  "
      f"index dtype={W64.col_indices().dtype}")
assert max(W64._nnz(), *W64.shape) < 2**31 - 1, "indices do not fit in int32"

W32 = torch.sparse_csr_tensor(
    W64.crow_indices().to(torch.int32),
    W64.col_indices().to(torch.int32),
    W64.values(), size=W64.shape, dtype=W64.dtype, device=W64.device)

bytes64 = (W64.shape[0] + 1) * 8 + W64._nnz() * 8 + W64._nnz() * 4
bytes32 = (W64.shape[0] + 1) * 4 + W64._nnz() * 4 + W64._nnz() * 4
print(f"  bytes touched per SpMV: int64 {bytes64/1e6:.1f} MB -> "
      f"int32 {bytes32/1e6:.1f} MB  ({bytes32/bytes64:.3f}x)")

g = torch.Generator(device="cuda").manual_seed(0)
n_probe, mism = 96, 0
for k in range(n_probe):
    nz = (2, 40, 500, 5000)[k % 4]
    s = torch.zeros(1, W64.shape[0], device="cuda")
    s[0, torch.randint(0, W64.shape[0], (nz,), device="cuda", generator=g)] = 1.0
    ref = torch.matmul(s, W64.transpose(0, 1))
    got = torch.matmul(s, W32.transpose(0, 1))
    if not torch.equal(ref, got):
        mism += 1
        if mism == 1:
            d = (ref - got).abs().max().item()
            print(f"  first mismatch at probe {k} (nz={nz}), max|delta|={d:.3e}")
print(f"  bit-identical on {n_probe - mism}/{n_probe} probes  "
      f"-> {'EXACT' if mism == 0 else 'NOT EXACT'}")

sp = brain.state.spikes
t64 = bench(lambda: torch.matmul(sp, W64.transpose(0, 1)))
t32 = bench(lambda: torch.matmul(sp, W32.transpose(0, 1)))
print(f"  int64 SpMV {t64:.4f} ms   int32 SpMV {t32:.4f} ms   "
      f"speedup {t64/t32:.2f}x  (projected 1.50x)")
print(f"  implied bandwidth: int64 {bytes64/t64/1e6:.0f} GB/s  "
      f"int32 {bytes32/t32/1e6:.0f} GB/s")

# ===================================================================== 2
rule("2. ring buffer vs torch.roll -- bit-exact?")
B, L, NN = 1, 19, W64.shape[0]
db = torch.randn(B, L, NN, device="cuda")

# upstream: read slot 0, roll left, write newest into slot -1
roll_buf = db.clone()
ring_buf = db.clone()
head = 0  # ring index of the OLDEST slot, i.e. what slot 0 means upstream

ok = True
for step in range(60):
    inp = torch.randn(B, NN, device="cuda")
    read_roll = roll_buf[:, 0, :].clone()
    roll_buf = torch.roll(roll_buf, shifts=-1, dims=1)
    roll_buf[:, -1, :] = inp

    read_ring = ring_buf[:, head, :].clone()
    ring_buf[:, head, :] = inp          # oldest slot becomes the newest entry
    head = (head + 1) % L

    if not torch.equal(read_roll, read_ring):
        print(f"  DIVERGED at step {step}")
        ok = False
        break
print(f"  reads identical over 60 steps: {'EXACT' if ok else 'NOT EXACT'}")

t_roll = bench(lambda: torch.roll(db, shifts=-1, dims=1))
buf2 = db.clone()
src = torch.randn(B, NN, device="cuda")
t_ring = bench(lambda: buf2[:, 3, :].copy_(src))
print(f"  torch.roll {t_roll:.4f} ms   ring write {t_ring:.4f} ms   "
      f"saves {t_roll - t_ring:.4f} ms/step "
      f"({(t_roll-t_ring)*50:.2f} ms per exchange)")

# ===================================================================== 3
rule("3. FlyGym self-contact pairs -- duplicated by an ordered-key bug?")
try:
    from m4_loop import build_sim
    sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
    mdl = sim.physics.model.ptr
    npair = mdl.npair
    print(f"  model has npair = {npair:,} explicit collision pairs")
    pairs = set()
    dup = 0
    for i in range(npair):
        a, b = int(mdl.pair_geom1[i]), int(mdl.pair_geom2[i])
        key = (min(a, b), max(a, b))
        if key in pairs:
            dup += 1
        pairs.add(key)
    print(f"  unique unordered pairs : {len(pairs):,}")
    print(f"  duplicate (A,B)+(B,A)  : {dup:,}")
    if dup > 0:
        print(f"  => CONFIRMED: {dup/npair*100:.0f}% of pair checks are redundant.")
        print(f"     Deduplicating would remove {dup:,} of {npair:,} narrow-phase tests.")
    else:
        print("  => REFUTED: no ordered-key duplication in the compiled model.")
    sim.close()
except Exception as e:
    print(f"  check failed: {type(e).__name__}: {e}")

rule("SUMMARY")
print(f"  int32 CSR      : {'EXACT' if mism == 0 else 'NOT EXACT'}, "
      f"{t64/t32:.2f}x on the SpMV")
print(f"  ring buffer    : {'EXACT' if ok else 'NOT EXACT'}, "
      f"saves {(t_roll-t_ring)*50:.1f} ms/exchange")
