#!/usr/bin/env python3
"""M6: let the fly live in a persistent multi-source world.

EVERYTHING HERE THAT SPEEDS THINGS UP WAS VERIFIED BIT-EXACT FIRST:
  FastBrain          int32 CSR indices + ring-buffer delay line
                     -> 500/500 identical steps, M1 anchor reproduced exactly
                        (17,393 spikes / 403 active), 1.20x
  memoise_observation  Fly.get_observation built once per physics state, not
                     twice -> max|delta| 0.0 in pose and qpos, 1.21x
  render off in-loop   qpos logged instead; video rendered offline afterwards.
                     Also removes FlyGym's 0.92 MB/frame in-RAM accumulation,
                     which OOMs a 12 GB VM in about 60 s of fly life.

NOT adopted: deduplicating FlyGym's 1,086 redundant collision pairs. Measured no
duplicated pair in contact over 3,000 walking steps, so the saving is unproven --
and if legs ever do touch, removing one would change the contact force. That is a
dynamics change.

WHAT THIS DOES NOT SHOW. The fly has no valence machinery: the connectome model
is static, the readout is one DNa02 left/right difference, and "aversive" is a
sign I impose in numpy. Nor does it forage in any meaningful sense -- depletion
is bookkeeping in the arena, invisible to the brain except through odour
concentration. Treat this as a persistent environment, not as behaviour.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import DT  # noqa: E402
from bridge import LEFT_SENSORS, RIGHT_SENSORS, BridgeParams, dn_to_action  # noqa: E402
from fast import FastBrain  # noqa: E402
from fastsim import memoise_observation, observation_stats  # noqa: E402
from m4_loop import calibrate  # noqa: E402
from world import make_world  # noqa: E402

OUT = REPO / "out"
OUT.mkdir(exist_ok=True)
NEU = json.loads((REPO / "neurons.v783.json").read_text())
TIMESTEP = 1e-4
SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACTS = [f"{s}{p}{seg}" for s in "LR" for p in "FMH" for seg in SEGMENTS]


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def build(arena, seed=0):
    from flygym import Camera, Fly
    from flygym.examples.locomotion import HybridTurningController
    fly = Fly(enable_olfaction=True, enable_adhesion=True, draw_adhesion=False,
              contact_sensor_placements=CONTACTS,
              spawn_pos=(0.0, 0.0, 0.25), spawn_orientation=(0.0, 0.0, 0.0))
    cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
                 targeted_fly_names=[fly.name], play_speed=1.0, fps=24,
                 window_size=(640, 480), play_speed_text=False, output_path=None)
    sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena,
                                  timestep=TIMESTEP, seed=seed)
    memoise_observation(sim)
    return sim, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0, help="fly life, seconds")
    ap.add_argument("--exchange-hz", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true",
                    help="NULL MODEL: brain gets odour-free ORN drive, so any "
                         "structure in the trajectory is drift, not taxis.")
    args = ap.parse_args()

    p = BridgeParams()
    p.exchange_hz = args.exchange_hz

    rule(f"M6  {args.seconds:.0f} s of fly life"
         + ("   [NULL MODEL: odour-blind]" if args.blind else ""))

    arena = make_world()
    print("world:")
    for s in arena.sources:
        print(f"  {s.label:7s} at ({s.pos[0]:+6.1f},{s.pos[1]:+6.1f})  "
              f"intensity={tuple(s.intensity)}  edible={s.edible}")
    print(f"  periodic cell {arena.period:.0f} mm, wind {tuple(arena._wind.round(2))}, "
          f"plume gain {arena.gain:.3g}")

    orn_l = NEU["ORN"]["left"]["root_ids"]
    orn_r = NEU["ORN"]["right"]["root_ids"]
    brain = FastBrain(batch=1, stim_ids=orn_l + orn_r, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(
        [NEU["DN"]["DNa02"]["left"]["root_ids"][0],
         NEU["DN"]["DNa02"]["right"]["root_ids"][0]]), device="cuda")
    li = torch.as_tensor(brain.idx(orn_l), device="cuda")
    ri = torch.as_tensor(brain.idx(orn_r), device="cuda")

    intercept, slope, r2, _, _ = calibrate(brain, dn_cols, p)
    p.dn_baseline_asym, p.dn_asym_scale = intercept, slope

    sim, cam = build(arena, seed=args.seed)
    obs, _ = sim.reset(seed=args.seed)
    arena.reset_world()

    exch_ms = 1000.0 / p.exchange_hz
    n_brain = int(round(exch_ms / DT))
    n_phys = int(round((exch_ms / 1000.0) / TIMESTEP))
    n_exch = int(args.seconds * p.exchange_hz)
    assert n_brain > 0 and n_phys > 0
    print(f"\nloop: {n_exch} exchanges x ({n_brain} brain + {n_phys} physics) steps")

    # warm the rate estimator at the spawn odour, as M4 does
    alpha = 1.0 - float(np.exp(-exch_ms / p.readout_tau_ms))
    rates = brain.zero_rates()

    def orn_from(obs_):
        if args.blind:
            # Same total drive, ZERO left/right contrast: the brain still runs,
            # the body still walks, but no odour information can reach the DNs.
            inten = np.asarray(obs_["odor_intensity"])[0]
            base = float(np.clip(0.5 * (inten[LEFT_SENSORS].mean()
                                        + inten[RIGHT_SENSORS].mean())
                                 * p.intensity_to_hz, 0, p.orn_rate_max))
            return base, base
        l, r, _, _ = _odour(obs_)
        return l, r

    def _odour(obs_):
        inten = np.asarray(obs_["odor_intensity"])[0]
        left = float(inten[LEFT_SENSORS].mean())
        right = float(inten[RIGHT_SENSORS].mean())
        tot = left + right
        contrast = (left - right) / (tot + 1e-12)
        amp = float(np.clip(contrast * p.contrast_gain, -1.0, 1.0))
        base = float(np.clip(0.5 * tot * p.intensity_to_hz, 0.0, p.orn_rate_max))
        return (float(np.clip(base * (1 + amp), 0, p.orn_rate_max)),
                float(np.clip(base * (1 - amp), 0, p.orn_rate_max)),
                contrast, 0.5 * tot)

    l_hz, r_hz = orn_from(obs)
    rates.zero_(); rates[:, li] = l_hz; rates[:, ri] = r_hz
    wc = torch.zeros(1, len(dn_cols), device="cuda")
    with torch.no_grad():
        for _ in range(3000):
            wc += brain.step(rates).index_select(1, dn_cols)
    ema = (wc / 0.3).cpu().numpy().ravel()

    tr = {k: [] for k in ("t", "x", "y", "dn_l", "dn_r", "bias", "contrast",
                          "odour", "reserve0", "reserve1")}
    qpos_log = []
    t0 = time.perf_counter()

    for ex in range(n_exch):
        l_hz, r_hz, contrast, mean_i = (*orn_from(obs), *_odour(obs)[2:]) \
            if args.blind else _odour(obs)
        rates.zero_(); rates[:, li] = l_hz; rates[:, ri] = r_hz

        counts = torch.zeros(1, len(dn_cols), device="cuda")
        with torch.no_grad():
            for _ in range(n_brain):
                counts += brain.step(rates).index_select(1, dn_cols)
        inst = (counts / (n_brain * DT / 1000.0)).cpu().numpy().ravel()
        ema = ema + alpha * (inst - ema)
        action, bias, _ = dn_to_action(float(ema[0]), float(ema[1]), p)

        arena.begin_exchange(n_phys)          # skip odour evals nobody reads
        for _ in range(n_phys):
            obs, _, term, trunc, _ = sim.step(action)

        xy = np.asarray(obs["fly"][0])[:2]
        arena.feed(xy, exch_ms / 1000.0)
        qpos_log.append(np.asarray(sim.physics.data.qpos).copy())

        for k, v in (("t", ex / p.exchange_hz), ("x", xy[0]), ("y", xy[1]),
                     ("dn_l", ema[0]), ("dn_r", ema[1]), ("bias", bias),
                     ("contrast", contrast), ("odour", mean_i),
                     ("reserve0", arena.reserve[0]), ("reserve1", arena.reserve[1])):
            tr[k].append(float(v))

        if ex % max(1, n_exch // 10) == 0:
            el = time.perf_counter() - t0
            done = (ex + 1) / n_exch
            print(f"  t={ex/p.exchange_hz:6.2f}s  pos=({xy[0]:+7.1f},{xy[1]:+7.1f})  "
                  f"odour={mean_i:.5f}  DN L/R={ema[0]:5.1f}/{ema[1]:4.1f}  "
                  f"bias={bias:+.2f}  reserves={arena.reserve.round(2)}  "
                  f"[{el/60:.1f}/{el/max(done,1e-9)/60:.1f} min]")
        if term or trunc:
            print(f"  terminated at exchange {ex}")
            break

    wall = time.perf_counter() - t0
    tr = {k: np.array(v) for k, v in tr.items()}
    st = observation_stats()

    rule("RESULT")
    print(f"  simulated {args.seconds:.1f} s in {wall/60:.2f} min  "
          f"-> {args.seconds/wall:.4f}x realtime")
    print(f"  (M4 baseline was 0.0137x; speedup {args.seconds/wall/0.0137:.2f}x)")
    print(f"  observation builds avoided: "
          f"{st['hits']/(st['builds']+st['hits'])*100:.0f}%")
    print(f"  path length: {np.sum(np.hypot(np.diff(tr['x']), np.diff(tr['y']))):.1f} mm")
    print(f"  net displacement: {np.hypot(tr['x'][-1]-tr['x'][0], tr['y'][-1]-tr['y'][0]):.1f} mm")
    print(f"\n  per-source residence (the metric that would show foraging):")
    for i, s in enumerate(arena.sources):
        print(f"    {s.label:7s} visits={arena.visits[i]:3d}  "
              f"dwell={arena.dwell_s[i]:6.2f} s  "
              f"consumed={arena.consumed[i]:.3f}  reserve={arena.reserve[i]:.3f}")
    print(f"\n  mean odour at antennae: {tr['odour'].mean():.6f} "
          f"(start {tr['odour'][0]:.6f} -> end {tr['odour'][-1]:.6f})")

    tag = "blind" if args.blind else "intact"
    np.savez_compressed(OUT / f"m6_{tag}.npz", qpos=np.array(qpos_log),
                        **{f"tr_{k}": v for k, v in tr.items()})
    (OUT / f"m6_{tag}_report.json").write_text(json.dumps({
        "seconds": args.seconds, "wall_min": wall / 60,
        "realtime_factor": args.seconds / wall,
        "speedup_vs_m4": args.seconds / wall / 0.0137,
        "blind": args.blind, "exchange_hz": args.exchange_hz,
        "obs_builds_avoided_pct": st["hits"] / (st["builds"] + st["hits"]) * 100,
        "path_mm": float(np.sum(np.hypot(np.diff(tr["x"]), np.diff(tr["y"])))),
        "sources": [{"label": s.label, "pos": list(s.pos),
                     "visits": int(arena.visits[i]),
                     "dwell_s": float(arena.dwell_s[i]),
                     "consumed": float(arena.consumed[i])}
                    for i, s in enumerate(arena.sources)],
        "mean_odour": float(tr["odour"].mean()),
        "calibration": {"intercept": intercept, "slope": slope, "r2": r2},
        "params": p.to_dict(),
    }, indent=2))
    print(f"\n  wrote out/m6_{tag}.npz and out/m6_{tag}_report.json")
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
