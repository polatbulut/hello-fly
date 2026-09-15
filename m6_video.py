#!/usr/bin/env python3
"""Render the living fly offline from the logged qpos trajectory.

WHY OFFLINE, AND WHY NOT FlyGym's Camera.
Rendering inside the loop cost ~158 ms per frame -- 43% of the whole closed
loop -- and FlyGym's Camera also accumulates ~0.92 MB per frame in RAM, which
OOMs a 12 GB VM after about 60 s of fly life. So the loop logs qpos (752 B per
frame) and the video is produced here, afterwards, off the critical path.

That ~158 ms is measured to be almost entirely fixed per-call overhead in
dm_control's physics.render(): it barely moves with resolution (155.6 ms at
320x240 vs 158.7 ms at 640x480), whereas a bare mujoco.Renderer scales properly
(7.7 -> 10.2 ms). So this uses mujoco.Renderer directly with a tracking overhead
camera, which is ~15x cheaper per frame and shows the world rather than a 6.6 mm
crop of the fly (FlyGym's camera_top at fovy=45 sits 8 mm up).

Replaying qpos is exact: qpos fully determines body poses via mj_forward, so the
frames show the trajectory that was actually simulated, not a re-simulation.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "out"

TAG = sys.argv[1] if len(sys.argv) > 1 else "intact"
FPS = 24
W, H = 640, 480
VIEW_MM = 90.0          # vertical extent of the frame at the ground plane


def main():
    z = np.load(OUT / f"m6_{TAG}.npz")
    rep = json.loads((OUT / f"m6_{TAG}_report.json").read_text())
    qpos = z["qpos"]
    x, y, t = z["tr_x"], z["tr_y"], z["tr_t"]
    print(f"{TAG}: {len(qpos)} logged frames over {rep['seconds']:.0f} s")

    # Rebuild the same model the run used, so the arena markers are present.
    from m6_live import build          # noqa: E402
    from world import make_world       # noqa: E402
    arena = make_world()
    sim, _ = build(arena, seed=0)
    sim.reset(seed=0)
    model = sim.physics.model.ptr
    data = sim.physics.data.ptr
    print(f"model nq={model.nq}  logged qpos width={qpos.shape[1]}")
    if qpos.shape[1] != model.nq:
        print("  qpos width does not match this model; aborting")
        sim.close()
        return 1

    # The offscreen framebuffer is capped by the model's <global offwidth/offheight>.
    try:
        renderer = mujoco.Renderer(model, height=H, width=W)
    except ValueError as e:
        print(f"  {e}\n  falling back to 640x480")
        renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 90.0
    cam.elevation = -89.0            # near-overhead, slight tilt for depth cues
    cam.distance = VIEW_MM

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    stride = max(1, int(round(len(qpos) / (rep["seconds"] * FPS))))
    idx = range(0, len(qpos), stride)
    print(f"rendering every {stride}th frame -> {len(list(idx))} frames at {FPS} fps")

    frames = []
    t0 = time.perf_counter()
    for k in idx:
        data.qpos[:] = qpos[k]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        cam.lookat[:] = (float(x[k]), float(y[k]), 0.0)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        frames.append(renderer.render().copy())
    wall = time.perf_counter() - t0
    print(f"  {wall:.1f} s for {len(frames)} frames "
          f"({wall/len(frames)*1000:.1f} ms/frame)")
    print(f"  vs FlyGym's in-loop Camera at ~158 ms/frame: "
          f"{158/(wall/len(frames)*1000):.0f}x cheaper")

    dst = OUT / f"m6_{TAG}_life.mp4"
    imageio.mimwrite(dst, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"  wrote {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
    renderer.close()
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
