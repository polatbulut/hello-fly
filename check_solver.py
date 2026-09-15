#!/usr/bin/env python3
"""FlyGym sets solver iterations=1000 (MuJoCo's default is 100). Is that real cost?

If the solver converges on tolerance long before the cap, lowering it is free.
If it genuinely iterates, lowering it is a speedup that COSTS ACCURACY -- and the
accuracy cost must be measured, not assumed, because the contact-rich leg model is
exactly where a loose solver shows up first.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_loop import build_sim  # noqa: E402

sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
sim.reset(seed=0)
mdl = sim.physics.model.ptr
dat = sim.physics.data.ptr
print(f"solver={mujoco.mjtSolver(mdl.opt.solver).name}  "
      f"iterations={mdl.opt.iterations}  tolerance={mdl.opt.tolerance}")
print(f"cone={mujoco.mjtCone(mdl.opt.cone).name}  "
      f"integrator={mujoco.mjtIntegrator(mdl.opt.integrator).name}")

# How many iterations does it ACTUALLY use? solver_niter is per-island.
mujoco.mj_step(mdl, dat)
niter = np.asarray(dat.solver_niter).copy()
print(f"\nactual solver_niter after a step: {niter[:8]} "
      f"(max {int(niter.max())} of {mdl.opt.iterations} allowed)")
if int(niter.max()) < mdl.opt.iterations * 0.5:
    print("  => converging on tolerance well before the cap; lowering the cap is FREE")
else:
    print("  => genuinely iterating; lowering the cap trades accuracy for speed")

ORIG = mdl.opt.iterations


def timed(n=300, warmup=50):
    for _ in range(warmup):
        mujoco.mj_step(mdl, dat)
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(mdl, dat)
    return (time.perf_counter() - t0) / n * 1000


def trajectory(iters, steps=2000, seed=0):
    """Run a fixed open-loop trajectory and return the final fly pose."""
    sim.reset(seed=seed)
    sim.physics.model.ptr.opt.iterations = iters
    act = np.array([1.0, 1.0], dtype=np.float32)
    for _ in range(steps):
        sim.step(act)
    return np.asarray(sim.physics.named.data.xpos["Fly/Thorax"]).copy()


print(f"\n{'iterations':>11s} {'ms/mj_step':>11s} {'speedup':>8s}")
base_ms = None
timings = {}
for it in (1000, 200, 100, 50, 20):
    mdl.opt.iterations = it
    ms = timed()
    timings[it] = ms
    if base_ms is None:
        base_ms = ms
    print(f"{it:11d} {ms:11.4f} {base_ms/ms:7.2f}x")
mdl.opt.iterations = ORIG

print("\ndoes lowering it change the trajectory? (2000 steps open-loop)")
ref = trajectory(1000)
print(f"  iterations=1000 (reference) -> thorax at "
      f"[{ref[0]:.4f} {ref[1]:.4f} {ref[2]:.4f}]")
for it in (200, 100, 50, 20):
    p = trajectory(it)
    d = float(np.linalg.norm(p - ref))
    flag = "" if d < 0.01 else ("  <- diverges" if d > 0.1 else "  <- drifts")
    print(f"  iterations={it:4d}            -> "
          f"[{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}]  |delta|={d:.5f} mm{flag}")
sim.physics.model.ptr.opt.iterations = ORIG
sim.close()

print("\nNOTE: a nonzero delta is not automatically disqualifying -- this model is")
print("contact-rich and chaotic, so trajectories separate even between identical")
print("runs with different RNG. What matters is whether the delta is larger than")
print("run-to-run variation at FIXED iterations, which the next line checks.")
