#!/usr/bin/env python3
"""Empirically introspect FlyGym 1.2.1's odor + turning API. Trust the objects, not the docs."""
import importlib.metadata as md
import inspect
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import flygym

print("versions")
for p in ("flygym", "mujoco", "gymnasium", "numpy", "dm_control"):
    try:
        print(f"  {p:12s} {md.version(p)}")
    except Exception as e:
        print(f"  {p:12s} ? ({e})")

print(f"\nflygym package attrs: {[a for a in dir(flygym) if not a.startswith('_')]}")


def sig(obj, name):
    try:
        print(f"\n{name}{inspect.signature(obj)}")
    except Exception as e:
        print(f"\n{name}(?) -- {e}")


print("\n" + "=" * 72)
print("ARENAS")
print("=" * 72)
import flygym.arena as arena
print(f"flygym.arena: {[a for a in dir(arena) if not a.startswith('_')]}")
for cls in ("OdorArena", "FlatTerrain", "BaseArena"):
    if hasattr(arena, cls):
        sig(getattr(arena, cls).__init__, f"arena.{cls}.__init__")

print("\n" + "=" * 72)
print("FLY / SIMULATION")
print("=" * 72)
for cls in ("Fly", "Camera", "SingleFlySimulation", "Simulation"):
    if hasattr(flygym, cls):
        sig(getattr(flygym, cls).__init__, f"flygym.{cls}.__init__")

print("\n" + "=" * 72)
print("TURNING CONTROLLER")
print("=" * 72)
try:
    from flygym.examples.locomotion import HybridTurningController
    sig(HybridTurningController.__init__, "HybridTurningController.__init__")
    sig(HybridTurningController.step, "HybridTurningController.step")
    print("  import path: flygym.examples.locomotion.HybridTurningController")
except Exception as e:
    print(f"  HybridTurningController: {e}")
try:
    from flygym.examples.locomotion import HybridTurningNMF
    sig(HybridTurningNMF.__init__, "HybridTurningNMF.__init__")
    print("  import path: flygym.examples.locomotion.HybridTurningNMF")
except Exception as e:
    print(f"  HybridTurningNMF: {e}")

import flygym.examples.locomotion as loco
print(f"\nflygym.examples.locomotion: {[a for a in dir(loco) if not a.startswith('_')]}")

print("\n" + "=" * 72)
print("LIVE TEST: build an odor arena + fly, step once, dump the obs dict")
print("=" * 72)
from flygym import Fly, Camera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController

odor_source = np.array([[24.0, 0.0, 1.5]])           # x, y, z
peak_intensity = np.array([[1.0, 0.0]])              # (n_sources, n_dims)
ar = OdorArena(odor_source=odor_source, peak_odor_intensity=peak_intensity,
               diffuse_func=lambda x: x ** -2)
print(f"arena: {type(ar).__name__}  n_sources={ar.odor_source.shape} "
      f"dims={getattr(ar, 'num_sensors', '?')}")

# HybridTurningController's stumble_segments default is ('Tibia','Tarsus1',
# 'Tarsus2'), and it hard-errors unless contact sensors exist on all of them.
# The Fly default placement covers Tarsus1-5 but NOT Tibia, so it must be added.
SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACTS = [f"{side}{pos}{seg}"
            for side in "LR" for pos in "FMH" for seg in SEGMENTS]
fly = Fly(enable_olfaction=True, enable_adhesion=True, draw_adhesion=True,
          contact_sensor_placements=CONTACTS,
          spawn_pos=(0, 0, 0.25))
print(f"fly: olfaction enabled, {len(fly.contact_sensor_placements)} contact sensors")

cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
             targeted_fly_names=[fly.name], play_speed=0.1)
sim = HybridTurningController(fly=fly, cameras=[cam], arena=ar, timestep=1e-4,
                              seed=0)
obs, info = sim.reset(seed=0)
print(f"\nreset() -> obs keys: {sorted(obs.keys())}")
for k, v in sorted(obs.items()):
    a = np.asarray(v)
    print(f"  {k:22s} shape={str(a.shape):14s} dtype={a.dtype}")

print(f"\naction_space : {sim.action_space}")
print(f"odor_intensity   = {np.asarray(obs['odor_intensity'])}")
print(f"  -> shape {np.asarray(obs['odor_intensity']).shape} "
      f"= (n_odor_dims, n_sensors)")
print(f"fly position     = {obs['fly'][0]}")

obs, rew, term, trunc, info = sim.step(np.array([1.0, 1.0]))
print(f"\nafter step([1,1]): odor={np.asarray(obs['odor_intensity']).ravel()} "
      f"pos={obs['fly'][0]}")
print(f"n sensors = {np.asarray(obs['odor_intensity']).shape[-1]}")
print(f"\nfly.get_olfaction? {hasattr(fly, 'get_olfaction')}")
print(f"arena.get_olfaction? {hasattr(ar, 'get_olfaction')}")
if hasattr(ar, "get_olfaction"):
    sig(ar.get_olfaction, "arena.OdorArena.get_olfaction")
print(f"sensor positions attr: {[a for a in dir(fly) if 'sensor' in a.lower()][:8]}")
print("\nOK")
