#!/usr/bin/env python3
"""Isolate the mjWARN_BADCTRL crash: which action, and does 50-steps-per-action matter?"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_loop import build_sim  # noqa: E402

SRC = ((22.0, 11.0, 1.5),)

for label, act, dtype in [
    ("[1.0,1.0] float64 (M2 baseline)", np.array([1.0, 1.0]), "f8"),
    ("[1.5,0.2] float64", np.array([1.5, 0.2]), "f8"),
    ("[1.5,0.2] float32", np.array([1.5, 0.2], dtype=np.float32), "f4"),
    ("[1.5,0.2] via np.clip+astype", np.clip([1.8, 0.2], -0.5, 1.5).astype(np.float32), "f4"),
]:
    sim, cam = build_sim(SRC, output_path=None, seed=0)
    obs, _ = sim.reset(seed=0)
    try:
        for i in range(200):
            obs, _, term, trunc, _ = sim.step(act)
        print(f"  OK    {label:38s} -> pos={np.asarray(obs['fly'][0])[:2]}")
    except Exception as e:
        print(f"  CRASH {label:38s} -> {type(e).__name__}: {str(e)[:90]}")
    sim.close()

# Does calling render() matter? M2 rendered every step; M4 renders every 50.
for label, every in [("render every step", 1), ("render every 50 steps", 50),
                     ("never render", 0)]:
    sim, cam = build_sim(SRC, output_path=None, seed=0)
    obs, _ = sim.reset(seed=0)
    try:
        for i in range(200):
            obs, _, term, trunc, _ = sim.step(np.array([1.0, 1.0]))
            if every and i % every == 0:
                sim.render()
        print(f"  OK    {label:38s}")
    except Exception as e:
        print(f"  CRASH {label:38s} -> {type(e).__name__}: {str(e)[:90]}")
    sim.close()

# Does reusing one sim across reset() cycles matter?
sim, cam = build_sim(SRC, output_path=None, seed=0)
for r in range(3):
    obs, _ = sim.reset(seed=0)
    try:
        for i in range(100):
            obs, _, term, trunc, _ = sim.step(np.array([1.2, 0.5]))
        print(f"  OK    reset cycle {r}")
    except Exception as e:
        print(f"  CRASH reset cycle {r} -> {type(e).__name__}: {str(e)[:90]}")
sim.close()
print("done")
