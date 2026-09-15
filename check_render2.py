#!/usr/bin/env python3
"""Corrected render benchmark: cost per FRAME ACTUALLY PRODUCED.

Camera.render() early-outs when not enough simulated time has elapsed
(`if curr_time < len(self._frames) * self._eff_render_interval: return None`),
so timing a tight loop of render() calls without stepping mostly times skips.
Here we step between renders and divide by frames genuinely produced.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_loop import build_sim  # noqa: E402
from flygym import Camera, Fly  # noqa: E402
from flygym.arena import OdorArena  # noqa: E402
from flygym.examples.locomotion import HybridTurningController  # noqa: E402

SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACTS = [f"{s}{p}{seg}" for s in "LR" for p in "FMH" for seg in SEGMENTS]


def make(draw_adhesion, play_speed_text, window=(640, 480)):
    arena = OdorArena(odor_source=np.array([[22.0, 12.0, 1.5]]),
                      peak_odor_intensity=np.array([[1.0, 0.0]]),
                      diffuse_func=lambda x: x ** -2)
    fly = Fly(enable_olfaction=True, enable_adhesion=True,
              draw_adhesion=draw_adhesion,
              contact_sensor_placements=CONTACTS, spawn_pos=(0, 0, 0.25))
    cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
                 targeted_fly_names=[fly.name], play_speed=0.15, fps=30,
                 window_size=window, play_speed_text=play_speed_text,
                 output_path=None)
    sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena,
                                  timestep=1e-4, seed=0)
    sim.reset(seed=0)
    return sim, cam


def measure(sim, cam, n_frames=25):
    """Step until n_frames have actually been rendered; return ms per frame."""
    act = np.array([1.0, 1.0], dtype=np.float32)
    # warm up one frame (first render builds the GL context)
    while len(cam._frames) < 2:
        sim.step(act)
        sim.render()

    start_frames = len(cam._frames)
    render_time = 0.0
    step_time = 0.0
    while len(cam._frames) - start_frames < n_frames:
        t0 = time.perf_counter()
        sim.step(act)
        t1 = time.perf_counter()
        sim.render()
        t2 = time.perf_counter()
        step_time += t1 - t0
        render_time += t2 - t1
    made = len(cam._frames) - start_frames
    return render_time / made * 1000, step_time, made


print(f"{'configuration':44s} {'ms/frame':>10s}")
for label, kw in [
    ("as used in M4 (adhesion+text, 640x480)", dict(draw_adhesion=True, play_speed_text=True)),
    ("no adhesion markers", dict(draw_adhesion=False, play_speed_text=True)),
    ("no adhesion, no text overlay", dict(draw_adhesion=False, play_speed_text=False)),
    ("no adhesion/text, 320x240", dict(draw_adhesion=False, play_speed_text=False, window=(320, 240))),
]:
    try:
        sim, cam = make(**kw)
        ms, st, made = measure(sim, cam)
        print(f"  {label:42s} {ms:10.2f}   ({made} frames)")
        sim.close()
    except Exception as e:
        print(f"  {label:42s} FAILED {type(e).__name__}: {str(e)[:50]}")

print()
print("frames are produced every play_speed/fps = 0.15/30 = 0.005 s of sim time,")
print("i.e. exactly once per 5 ms exchange at the current settings.")
print("Lowering play_speed or fps renders fewer frames per simulated second.")
