#!/usr/bin/env python3
"""Does memoising get_observation change the trajectory? It must not.

Run the same open-loop walk with and without the patch and compare the fly's
pose step by step. Anything but an exact match disqualifies it.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from m4_loop import build_sim  # noqa: E402
from fastsim import (  # noqa: E402
    memoise_observation, observation_stats, unmemoise_observation,
)

STEPS = 1500
ACT = np.array([1.0, 1.0], dtype=np.float32)


def run(patch, steps=STEPS):
    sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
    if patch:
        memoise_observation(sim)
    else:
        unmemoise_observation(sim)   # the patch is class-level; undo it for A/B
    sim.reset(seed=0)
    poses = []
    t0 = time.perf_counter()
    for i in range(steps):
        obs, *_ = sim.step(ACT)
        if i % 50 == 0:
            poses.append(np.asarray(obs["fly"][0]).copy())
    wall = time.perf_counter() - t0
    qpos = np.asarray(sim.physics.data.qpos).copy()
    sim.close()
    return np.array(poses), qpos, wall


print(f"running {STEPS} open-loop steps, with and without the patch")
p_ref, q_ref, t_ref = run(False)
p_opt, q_opt, t_opt = run(True)

print(f"\n  reference : {t_ref:6.2f} s  ({t_ref/STEPS*1000:.3f} ms/step)")
print(f"  memoised  : {t_opt:6.2f} s  ({t_opt/STEPS*1000:.3f} ms/step)   "
      f"{t_ref/t_opt:.2f}x")
st = observation_stats()
print(f"  observation builds={st['builds']}  cache hits={st['hits']}  "
      f"({st['hits']/(st['builds']+st['hits'])*100:.0f}% avoided)")

dp = np.abs(p_ref - p_opt).max()
dq = np.abs(q_ref - q_opt).max()
print(f"\n  max |delta| in sampled fly pose : {dp:.3e}")
print(f"  max |delta| in final qpos       : {dq:.3e}")
exact = dp == 0.0 and dq == 0.0
print(f"  => {'BIT-EXACT' if exact else 'NOT EXACT -- do not adopt'}")

if not exact:
    print("\n  NOTE: this model is contact-rich and chaotic, so a nonzero delta")
    print("  could in principle be benign. It is not treated as benign here:")
    print("  the two runs are deterministic and identically seeded, so any")
    print("  difference at all means the patch altered the computation.")

saving = (t_ref - t_opt) / STEPS * 1000 * 50
print(f"\n  projected saving per 50-step exchange: {saving:.1f} ms")
