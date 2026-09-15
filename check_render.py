#!/usr/bin/env python3
"""Is MuJoCo's EGL render actually on the GPU, or silently on llvmpipe?

99.57 ms for one 640x480 offscreen frame is ~10x slower than a GPU EGL render
should be, which is the signature of a software rasteriser fallback.
"""
import os
import sys
import time

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "egl"
os.environ["MUJOCO_GL"] = BACKEND
os.environ.setdefault("PYOPENGL_PLATFORM", BACKEND)

import numpy as np
import mujoco

print(f"MUJOCO_GL = {os.environ['MUJOCO_GL']}")
print(f"mujoco    = {mujoco.__version__}")

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)

for w, h in ((320, 240), (640, 480), (1280, 960)):
    try:
        r = mujoco.Renderer(m, height=h, width=w)
        mujoco.mj_forward(m, d)
        for _ in range(3):
            r.update_scene(d)
            r.render()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            r.update_scene(d)
            img = r.render()
        ms = (time.perf_counter() - t0) / n * 1000
        px = w * h
        print(f"  {w}x{h:4d}  {ms:7.2f} ms/frame   "
              f"{px/ms/1000:8.1f} Mpix/s")
        r.close()
    except Exception as e:
        print(f"  {w}x{h}: FAILED {type(e).__name__}: {e}")

# What is the GL implementation actually being used?
print("\nGL implementation:")
try:
    from OpenGL import GL
    for name, enum in (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                       ("version", GL.GL_VERSION)):
        try:
            v = GL.glGetString(enum)
            print(f"  {name:9s}: {v.decode() if v else '(null)'}")
        except Exception as e:
            print(f"  {name:9s}: {type(e).__name__}")
except Exception as e:
    print(f"  pyopengl unavailable: {e}")

print("\ninterpretation:")
print("  'NVIDIA ...'            -> real GPU path")
print("  'llvmpipe' / 'softpipe' -> SOFTWARE rasteriser, the 100 ms is explained")
print("  'D3D12 (NVIDIA ...)'    -> WSL's mesa-on-D3D12 bridge, GPU but with overhead")
