#!/usr/bin/env python3
"""Decompose FlyGym's 3.464 ms step. How much is MuJoCo, how much is the wrapper?

Physics is 53.5% of the closed loop, so this single number decides whether the
loop is optimisable at all.
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


def bench(fn, n=300, warmup=50):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
obs, _ = sim.reset(seed=0)
act = np.array([1.0, 1.0], dtype=np.float32)

phys = sim.physics                      # dm_control Physics
mdl = phys.model.ptr                    # raw MjModel
dat = phys.data.ptr                     # raw MjData
print(f"model: nq={mdl.nq} nv={mdl.nv} nu={mdl.nu} nbody={mdl.nbody} "
      f"ngeom={mdl.ngeom} nsensor={mdl.nsensor}")
print(f"timestep: {mdl.opt.timestep}  solver iterations: {mdl.opt.iterations}")
print()

rows = []
rows.append(("sim.step(action)  [full FlyGym]", bench(lambda: sim.step(act), n=200)))
rows.append(("phys.step()       [dm_control]", bench(lambda: phys.step(), n=300)))
rows.append(("mujoco.mj_step()  [raw MuJoCo]", bench(lambda: mujoco.mj_step(mdl, dat), n=300)))

# What does FlyGym add on top? Observation building is the suspect: it assembles
# odour, 36x3 contact forces, 3x42 joints, end effectors and cardinal vectors --
# every single step, when the loop only reads two of those and only once per exchange.
try:
    fly = sim.flies[0] if hasattr(sim, "flies") else sim.fly
    rows.append(("fly.get_observation(sim)", bench(lambda: fly.get_observation(sim), n=200)))
except Exception as e:
    print(f"  (get_observation not benchmarkable directly: {type(e).__name__}: {e})")

try:
    rows.append(("arena.get_olfaction(...)",
                 bench(lambda: sim.arena.get_olfaction(
                     fly.get_olfaction_sensor_positions(sim.physics)
                     if hasattr(fly, "get_olfaction_sensor_positions")
                     else np.zeros((4, 3))), n=200)))
except Exception as e:
    print(f"  (olfaction not benchmarkable: {type(e).__name__})")

print(f"{'what':40s} {'ms':>9s}   {'x raw mj_step':>14s}")
raw = [v for k, v in rows if "raw MuJoCo" in k][0]
for k, v in rows:
    print(f"  {k:38s} {v:9.4f}   {v/raw:13.1f}x")

full = [v for k, v in rows if "full FlyGym" in k][0]
print()
print(f"  raw MuJoCo physics is {raw/full*100:.1f}% of FlyGym's step")
print(f"  FlyGym wrapper overhead: {full-raw:.3f} ms per step "
      f"({(full-raw)/full*100:.1f}%)")
print()
print(f"  At 50 steps per exchange that wrapper overhead is "
      f"{(full-raw)*50:.1f} ms per 5 ms of simulated time.")
print(f"  If only 1 of 50 steps needed a full observation, the saving would be")
print(f"  roughly {(full-raw)*49:.1f} ms per exchange.")

sim.close()
