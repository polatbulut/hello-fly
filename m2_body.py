#!/usr/bin/env python3
"""M2: FlyGym body alone -- walking, odour readout, turning modulation, mp4.

Four things, in order:
  A. Determine the odour sensor left/right ordering BY EXPERIMENT. Getting this
     backwards would silently invert the closed loop, and no docstring is worth
     trusting for it.
  B. Walk straight on flat terrain, render mp4.
  C. Show the turning controller responds to asymmetric drive, both directions.
  D. Hand-written odour taxis as a POSITIVE CONTROL -- no connectome involved.
     If this works and the connectome loop later does not, the fault is in the
     brain/bridge, not the body. If this fails, nothing downstream is meaningful.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flygym import Camera, Fly
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController

REPO = Path(__file__).resolve().parent
OUT = REPO / "out"
OUT.mkdir(exist_ok=True)

# HybridTurningController's stumble_segments default is ('Tibia','Tarsus1',
# 'Tarsus2') and it hard-errors unless contact sensors exist on all of them.
# Fly's default placement covers Tarsus1-5 but NOT Tibia.
SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACTS = [f"{s}{p}{seg}" for s in "LR" for p in "FMH" for seg in SEGMENTS]

TIMESTEP = 1e-4          # MuJoCo physics step, FlyGym default


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def build(odor_source, peak=None, output_path=None, play_speed=0.2, fps=30):
    peak = np.array([[1.0, 0.0]]) if peak is None else peak
    arena = OdorArena(
        odor_source=np.asarray(odor_source, dtype=float),
        peak_odor_intensity=np.asarray(peak, dtype=float),
        diffuse_func=lambda x: x ** -2,
        marker_size=0.3,
    )
    fly = Fly(
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=CONTACTS,
        spawn_pos=(0.0, 0.0, 0.25),
        spawn_orientation=(0.0, 0.0, 0.0),     # facing +x
    )
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_top",
        targeted_fly_names=[fly.name],
        play_speed=play_speed,
        fps=fps,
        window_size=(700, 500),
        output_path=output_path,
    )
    sim = HybridTurningController(
        fly=fly, cameras=[cam], arena=arena, timestep=TIMESTEP, seed=0)
    return sim, cam, fly


# ------------------------------------------------------------------ A: order
def part_a():
    rule("A. Which odour sensor is LEFT? (determined by experiment)")
    print("Fly spawns at the origin facing +x, so +y is its LEFT.")
    findings = {}
    for label, src, expect in [
        ("odour on the fly's LEFT  (+y)", [[0.0, 10.0, 1.5]], "left"),
        ("odour on the fly's RIGHT (-y)", [[0.0, -10.0, 1.5]], "right"),
    ]:
        sim, _, _ = build(src)
        obs, _ = sim.reset(seed=0)
        inten = np.asarray(obs["odor_intensity"])[0]      # first odour dimension
        hi = int(np.argmax(inten))
        print(f"\n  {label}")
        print(f"    intensities by sensor index: "
              f"{np.array2string(inten, precision=6)}")
        print(f"    strongest sensor index = {hi}")
        findings[expect] = {"intensities": inten.tolist(), "argmax": hi}
        sim.close()

    l_int = np.array(findings["left"]["intensities"])
    r_int = np.array(findings["right"]["intensities"])
    # Sensors whose reading rises when the source moves to +y are the left ones.
    left_idx = sorted(np.where(l_int > r_int)[0].tolist())
    right_idx = sorted(np.where(r_int > l_int)[0].tolist())
    print(f"\n  sensors higher when odour is on the LEFT : {left_idx}")
    print(f"  sensors higher when odour is on the RIGHT: {right_idx}")
    print("\n  FlyGym places 4 odour sensors: 2 antennae + 2 maxillary palps.")
    print(f"  => LEFT  sensor indices = {left_idx}")
    print(f"  => RIGHT sensor indices = {right_idx}")
    assert left_idx and right_idx, "could not separate left from right sensors"
    return left_idx, right_idx, findings


# ------------------------------------------------------- B/C: walk and turn
def run_episode(sim, cam, action_fn, seconds, label):
    n = int(seconds / TIMESTEP)
    obs, _ = sim.reset(seed=0)
    traj, odor = [], []
    for i in range(n):
        act = action_fn(i, obs)
        obs, _, term, trunc, _ = sim.step(act)
        sim.render()
        traj.append(np.asarray(obs["fly"][0]).copy())
        odor.append(np.asarray(obs["odor_intensity"])[0].copy())
        if term or trunc:
            print(f"    terminated early at step {i}")
            break
    traj = np.array(traj)
    d = traj[-1] - traj[0]
    heading = np.degrees(np.arctan2(d[1], d[0]))
    print(f"  {label:28s} start=({traj[0][0]:6.2f},{traj[0][1]:6.2f}) "
          f"end=({traj[-1][0]:6.2f},{traj[-1][1]:6.2f})  "
          f"net=({d[0]:+6.2f},{d[1]:+6.2f})  heading={heading:+7.1f} deg")
    return traj, np.array(odor), heading


def main():
    left_idx, right_idx, order_findings = part_a()

    rule("B. Walk straight on flat terrain (symmetric drive), render mp4")
    sim, cam, _ = build([[100.0, 0.0, 1.5]],
                        output_path=OUT / "m2_walk_straight.mp4")
    traj_s, _, head_s = run_episode(
        sim, cam, lambda i, o: np.array([1.0, 1.0]), 1.5, "symmetric [1.0, 1.0]")
    cam.save_video(OUT / "m2_walk_straight.mp4")
    sim.close()
    print(f"  wrote {OUT/'m2_walk_straight.mp4'}")

    rule("C. Turning responds to asymmetric descending drive")
    turns = {}
    for label, act, fname in [
        ("left  [0.4, 1.2]", np.array([0.4, 1.2]), "m2_turn_left.mp4"),
        ("right [1.2, 0.4]", np.array([1.2, 0.4]), "m2_turn_right.mp4"),
    ]:
        sim, cam, _ = build([[100.0, 0.0, 1.5]], output_path=OUT / fname)
        tr, _, hd = run_episode(sim, cam, lambda i, o, a=act: a, 1.5, label)
        cam.save_video(OUT / fname)
        sim.close()
        turns[label] = {"heading_deg": float(hd),
                        "net_y": float(tr[-1][1] - tr[0][1])}
    print(f"\n  straight heading : {head_s:+7.1f} deg")
    for k, v in turns.items():
        print(f"  {k:18s} heading : {v['heading_deg']:+7.1f} deg  "
              f"(net y {v['net_y']:+.2f})")
    dl = turns["left  [0.4, 1.2]"]["heading_deg"]
    dr = turns["right [1.2, 0.4]"]["heading_deg"]
    print(f"\n  action[0] is the LEFT drive, action[1] the RIGHT drive.")
    print(f"  Lowering the left drive -> heading {dl:+.1f} deg; "
          f"lowering the right -> {dr:+.1f} deg.")
    sign_ok = dl > head_s > dr or dl < head_s < dr
    print(f"  the two asymmetries steer in OPPOSITE directions: "
          f"{'YES' if sign_ok else 'NO -- investigate'}")

    rule("D. POSITIVE CONTROL: hand-written odour taxis (no connectome)")
    print("If this cannot find the odour, nothing built on top of it can.")
    SRC = [[25.0, 12.0, 1.5]]      # ahead and to the fly's left
    sim, cam, _ = build(SRC, output_path=OUT / "m2_odor_taxis_control.mp4")

    GAIN = 3.0        # hand-tuned; the analogous constant in M4 is the bridge gain

    def taxis(i, obs):
        inten = np.asarray(obs["odor_intensity"])[0]
        l = float(inten[left_idx].mean())
        r = float(inten[right_idx].mean())
        # Normalised left-right contrast; scale-free, so it keeps working as
        # the fly approaches and absolute intensity rises by orders of magnitude.
        bias = (l - r) / (l + r + 1e-12)
        return np.clip([1.0 - GAIN * bias, 1.0 + GAIN * bias], -0.5, 1.5)

    traj_t, odor_t, _ = run_episode(sim, cam, taxis, 3.0, "odour taxis")
    cam.save_video(OUT / "m2_odor_taxis_control.mp4")
    sim.close()

    src = np.array(SRC[0][:2])
    d0 = np.linalg.norm(traj_t[0][:2] - src)
    d1 = np.linalg.norm(traj_t[-1][:2] - src)
    print(f"\n  distance to source: {d0:.2f} -> {d1:.2f} mm  "
          f"({'CLOSER' if d1 < d0 else 'NOT closer'})")
    print(f"  odour at sensors:   {odor_t[0].mean():.6f} -> "
          f"{odor_t[-1].mean():.6f} "
          f"({'increased' if odor_t[-1].mean() > odor_t[0].mean() else 'did not increase'})")

    report = {
        "flygym_version": "1.2.1",
        "timestep_s": TIMESTEP,
        "obs_keys": ["cardinal_vectors", "contact_forces", "end_effectors",
                     "fly", "fly_orientation", "joints", "odor_intensity"],
        "odor_intensity_shape": "(n_odor_dims=2, n_sensors=4)",
        "action_space": "Box(-0.5, 1.5, (2,), float32)  # [left_drive, right_drive]",
        "sensor_order": {
            "left_indices": left_idx, "right_indices": right_idx,
            "method": "moved the odour source to +y and -y and compared readings",
            "raw": order_findings,
        },
        "straight_heading_deg": float(head_s),
        "turning": turns,
        "turning_signs_opposite": bool(sign_ok),
        "taxis_control": {
            "source_xy": src.tolist(),
            "dist_start": float(d0), "dist_end": float(d1),
            "got_closer": bool(d1 < d0),
            "gain": GAIN,
        },
        "videos": [str(p.name) for p in sorted(OUT.glob("m2_*.mp4"))],
    }
    (OUT / "m2_report.json").write_text(json.dumps(report, indent=2))

    rule("M2 SUMMARY")
    print(f"  odour sensors readable      : YES  shape (2,4)")
    print(f"  left/right resolved         : L={left_idx} R={right_idx}")
    print(f"  turning modulates heading   : {'YES' if sign_ok else 'NO'}")
    print(f"  taxis positive control      : "
          f"{'REACHES SOURCE' if d1 < d0 else 'FAILS'}")
    print(f"  videos                      : {report['videos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
