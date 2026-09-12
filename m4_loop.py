#!/usr/bin/env python3
"""M4: close the loop. Odour -> ORNs -> connectome -> DNa02 -> turning -> physics.

RATE DECOUPLING
Physics runs at its native 1e-4 s step and the brain at its native 1e-4 s step,
but they exchange information only at `exchange_hz` (200 Hz = every 5 ms = 50
native steps each). They are NOT stepped in lockstep: between exchanges each
subsystem advances independently on its own clock, with the other's state held
constant. This is the standard quasi-static coupling assumption and it is valid
while 5 ms is short relative to both the DN membrane time constant (~20 ms) and
the stride period (~80 ms at 12 Hz).

WHY THE FULL 139k BRAIN, NOT A SUBSET
Measured in M1: 115 ms of simulated brain time per second of wall clock at
batch=1, using 1.45 GB of 11 GB VRAM. A 2.5 s episode therefore costs ~22 s of
brain compute -- entirely affordable. Subsetting to the ORN->DN pathway (hop<=3,
~42,400 neurons) would buy maybe 3x, at the cost of discarding ~11% of DNa02's
input synapses and introducing a second model whose divergence from the full
network would itself need validating. The measured numbers say we do not need
that trade, so we do not take it.

(Also measured in M1, and the reason batch stays at 1: batching is
counterproductive on this GPU. batch=4 takes 8x LONGER in absolute wall clock
than batch=1 for the same simulated duration, because matmul(dense, sparse_csr.T)
has no efficient batched path. Aggregate throughput FALLS from 115 to 58.)
"""
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
from brain import DT, Brain  # noqa: E402
from bridge import (  # noqa: E402
    LEFT_SENSORS, RIGHT_SENSORS, BridgeParams, dn_to_action, odour_to_orn_rates,
)

OUT = REPO / "out"
OUT.mkdir(exist_ok=True)
NEURONS = json.loads((REPO / "neurons.v783.json").read_text())

TIMESTEP = 1e-4
SEGMENTS = ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
CONTACTS = [f"{s}{p}{seg}" for s in "LR" for p in "FMH" for seg in SEGMENTS]


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def build_sim(odor_source, output_path=None, seed=0):
    from flygym import Camera, Fly
    from flygym.arena import OdorArena
    from flygym.examples.locomotion import HybridTurningController

    arena = OdorArena(
        odor_source=np.asarray(odor_source, dtype=float),
        peak_odor_intensity=np.array([[1.0, 0.0]]),
        diffuse_func=lambda x: x ** -2,
        marker_size=0.3,
    )
    fly = Fly(enable_olfaction=True, enable_adhesion=True, draw_adhesion=True,
              contact_sensor_placements=CONTACTS,
              spawn_pos=(0.0, 0.0, 0.25), spawn_orientation=(0.0, 0.0, 0.0))
    cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_top",
                 targeted_fly_names=[fly.name], play_speed=0.15, fps=30,
                 window_size=(640, 480), output_path=output_path)
    sim = HybridTurningController(fly=fly, cameras=[cam], arena=arena,
                                  timestep=TIMESTEP, seed=seed)
    return sim, cam


# --------------------------------------------------------------------- calib
def calibrate(brain, dn_cols, p: BridgeParams, ms=500.0, trials=2):
    """Fit (L-R) DNa02 against imposed ORN contrast. Returns (intercept, slope).

    This replaces two guessed constants with measured ones. The intercept is the
    model's fixed left-bias pedestal; the slope is its odour-contrast gain.
    """
    rule("CALIBRATION: DNa02 (L-R) vs imposed ORN contrast")
    orn_l = NEURONS["ORN"]["left"]["root_ids"]
    orn_r = NEURONS["ORN"]["right"]["root_ids"]
    li = torch.as_tensor(brain.idx(orn_l), device="cuda")
    ri = torch.as_tensor(brain.idx(orn_r), device="cuda")
    base = p.orn_rate_max / 2.0
    n_steps = int(ms / DT)

    xs, ys = [], []
    for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
        per = []
        for t in range(trials):
            rates = brain.zero_rates()
            rates[:, li] = base * (1.0 + c)
            rates[:, ri] = base * (1.0 - c)
            brain.reset(seed=7000 + t)
            counts = torch.zeros(1, len(dn_cols), device="cuda")
            with torch.no_grad():
                for _ in range(n_steps):
                    counts += brain.step(rates).index_select(1, dn_cols)
            torch.cuda.synchronize()
            cc = counts.cpu().numpy().ravel() / (ms / 1000.0)   # -> Hz
            per.append(cc[0] - cc[1])          # DNa02_left - DNa02_right
        xs.append(c)
        ys.append(float(np.mean(per)))
        print(f"  contrast {c:+.2f} -> L-R = {np.mean(per):+7.2f} Hz "
              f"(trials {[round(v, 1) for v in per]})")

    slope, intercept = np.polyfit(xs, ys, 1)
    pred = np.polyval([slope, intercept], xs)
    ss_res = float(((np.array(ys) - pred) ** 2).sum())
    ss_tot = float(((np.array(ys) - np.mean(ys)) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    print(f"\n  fit: (L-R) = {slope:+.2f} * contrast {intercept:+.2f}   R^2={r2:.3f}")
    print(f"  intercept {intercept:+.2f} Hz is the FIXED model left-bias pedestal")
    print(f"  slope     {slope:+.2f} Hz per unit contrast is the usable signal")
    return float(intercept), float(slope), float(r2), xs, ys


# ---------------------------------------------------------------------- loop
def run_closed_loop(p: BridgeParams, seconds=2.5, odor_source=((22.0, 11.0, 1.5),),
                    silence_ids=None, label="m4", seed=0, brain=None,
                    calib=None, render=True):
    """Run the closed loop. Returns a dict of traces."""
    orn_l = NEURONS["ORN"]["left"]["root_ids"]
    orn_r = NEURONS["ORN"]["right"]["root_ids"]
    dn_ids = [NEURONS["DN"]["DNa02"]["left"]["root_ids"][0],
              NEURONS["DN"]["DNa02"]["right"]["root_ids"][0],
              NEURONS["DN"]["DNa01"]["left"]["root_ids"][0],
              NEURONS["DN"]["DNa01"]["right"]["root_ids"][0]]
    dn_names = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R"]

    if brain is None:
        brain = Brain(batch=1, stim_ids=orn_l + orn_r, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(dn_ids), device="cuda")
    li = torch.as_tensor(brain.idx(orn_l), device="cuda")
    ri = torch.as_tensor(brain.idx(orn_r), device="cuda")

    silenced = brain.silence(silence_ids) if silence_ids else brain.unsilence()

    if calib is not None:
        p.dn_baseline_asym, p.dn_asym_scale = calib

    # UNITS: DT is in MILLISECONDS (0.1), TIMESTEP is in SECONDS (1e-4).
    # Mixing them silently rounded steps_per_exchange to 0, which made the brain
    # never step and turned counts/win_s into 0/0 = NaN. Keep the *1000.
    exchange_ms = 1000.0 / p.exchange_hz                              # 5.0 ms
    steps_per_exchange = int(round(exchange_ms / DT))                 # 50
    phys_per_exchange = int(round((exchange_ms / 1000.0) / TIMESTEP))  # 50
    assert steps_per_exchange > 0 and phys_per_exchange > 0, (
        f"bad step counts: brain={steps_per_exchange} phys={phys_per_exchange}")
    n_exchanges = int(seconds * p.exchange_hz)

    out_mp4 = OUT / f"{label}_fly.mp4" if render else None
    sim, cam = build_sim(odor_source, output_path=out_mp4, seed=seed)
    obs, _ = sim.reset(seed=seed)
    brain.reset(seed=seed)

    # Preallocate the DN spike raster on the GPU; transfer once at the end so
    # the loop never syncs on a per-step host copy.
    total_steps = n_exchanges * steps_per_exchange
    raster = torch.zeros(total_steps, len(dn_cols), device="cuda")

    rates = brain.zero_rates()

    # --- rate estimation ---------------------------------------------------
    # A 5 ms exchange window sees ~0.3 spikes from a 60 Hz neuron, so the
    # per-window estimate is 0 Hz or 200 Hz and is useless on its own. Keep an
    # exponential moving average with time constant readout_tau_ms, and SEED it
    # by running the brain at the initial odour condition first -- otherwise the
    # first ~tau of the episode is a transient in which the DN estimate climbs
    # from zero and the bias sits pinned at its clip.
    alpha = 1.0 - float(np.exp(-(1000.0 / p.exchange_hz) / p.readout_tau_ms))
    l_hz, r_hz, _, _ = odour_to_orn_rates(obs["odor_intensity"], p)
    rates.zero_()
    rates[:, li] = l_hz
    rates[:, ri] = r_hz
    warm_ms = 300.0
    warm_steps = int(warm_ms / DT)
    wc = torch.zeros(1, len(dn_cols), device="cuda")
    with torch.no_grad():
        for _ in range(warm_steps):
            wc += brain.step(rates).index_select(1, dn_cols)
    ema = (wc / (warm_ms / 1000.0)).cpu().numpy().ravel()
    print(f"    warmup {warm_ms:.0f} ms -> DNa02 L/R = {ema[0]:.1f}/{ema[1]:.1f} Hz "
          f"(EMA tau={p.readout_tau_ms:.0f} ms, alpha={alpha:.4f})")

    tr = {k: [] for k in ("t", "x", "y", "odor_l", "odor_r", "contrast",
                          "orn_l", "orn_r", "dn_l", "dn_r", "bias",
                          "act_l", "act_r", "dist")}
    src = np.array(odor_source[0][:2], dtype=float)
    t0 = time.perf_counter()
    k = 0
    for ex in range(n_exchanges):
        # ---- 1. sensory encoding -----------------------------------------
        l_hz, r_hz, contrast, mean_i = odour_to_orn_rates(obs["odor_intensity"], p)
        rates.zero_()
        rates[:, li] = l_hz
        rates[:, ri] = r_hz

        # ---- 2. brain advances 5 ms on its own clock ---------------------
        counts = torch.zeros(1, len(dn_cols), device="cuda")
        with torch.no_grad():
            for _ in range(steps_per_exchange):
                s = brain.step(rates)
                d = s.index_select(1, dn_cols)
                raster[k] = d[0]
                counts += d
                k += 1
        win_s = steps_per_exchange * DT / 1000.0
        inst = (counts / win_s).cpu().numpy().ravel()    # one sync per exchange
        ema = ema + alpha * (inst - ema)                 # smooth; see above
        dn_l, dn_r = float(ema[0]), float(ema[1])

        # ---- 3. motor decoding -------------------------------------------
        action, bias, asym = dn_to_action(dn_l, dn_r, p)
        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"non-finite action {action} at exchange {ex}: "
                f"dn_l={dn_l} dn_r={dn_r} bias={bias} asym={asym} "
                f"inst={inst} ema={ema}")

        # ---- 4. physics advances 5 ms on its own clock -------------------
        for _ in range(phys_per_exchange):
            obs, _, term, trunc, _ = sim.step(action)
        if render:
            sim.render()

        inten = np.asarray(obs["odor_intensity"])[0]
        xy = np.asarray(obs["fly"][0])[:2]
        for key, val in (("t", ex / p.exchange_hz), ("x", xy[0]), ("y", xy[1]),
                         ("odor_l", inten[LEFT_SENSORS].mean()),
                         ("odor_r", inten[RIGHT_SENSORS].mean()),
                         ("contrast", contrast), ("orn_l", l_hz), ("orn_r", r_hz),
                         ("dn_l", dn_l), ("dn_r", dn_r), ("bias", bias),
                         ("act_l", float(action[0])), ("act_r", float(action[1])),
                         ("dist", float(np.linalg.norm(xy - src)))):
            tr[key].append(float(val))

        if ex % max(1, n_exchanges // 8) == 0:
            print(f"    t={ex/p.exchange_hz:5.2f}s  pos=({xy[0]:6.2f},{xy[1]:6.2f})  "
                  f"contrast={contrast:+.4f}  ORN L/R={l_hz:5.1f}/{r_hz:5.1f}  "
                  f"DNa02 L/R={dn_l:5.1f}/{dn_r:5.1f}  bias={bias:+.3f}")
        if term or trunc:
            break

    wall = time.perf_counter() - t0
    tr = {key: np.array(v) for key, v in tr.items()}
    raster_np = raster[:k].cpu().numpy()
    if render:
        cam.save_video(out_mp4)
    sim.close()

    print(f"\n  {seconds:.1f} s simulated in {wall:.1f} s wall "
          f"({seconds/wall:.3f}x realtime)")
    print(f"  distance to source: {tr['dist'][0]:.2f} -> {tr['dist'][-1]:.2f} mm")
    return {"trace": tr, "raster": raster_np, "dn_names": dn_names,
            "wall_s": wall, "silenced": list(silenced) if silence_ids else [],
            "video": str(out_mp4) if render else None, "params": p.to_dict()}


def main():
    p = BridgeParams()
    orn_all = (NEURONS["ORN"]["left"]["root_ids"]
               + NEURONS["ORN"]["right"]["root_ids"])
    rule("M4  closing the loop")
    brain = Brain(batch=1, stim_ids=orn_all, device="cuda")
    dn_cols = torch.as_tensor(brain.idx([
        NEURONS["DN"]["DNa02"]["left"]["root_ids"][0],
        NEURONS["DN"]["DNa02"]["right"]["root_ids"][0]]), device="cuda")

    intercept, slope, r2, xs, ys = calibrate(brain, dn_cols, p)
    p.dn_baseline_asym, p.dn_asym_scale = intercept, slope

    rule("CLOSED LOOP -- mirrored pair (the decisive odour-dependence test)")
    print(f"  exchange {p.exchange_hz:.0f} Hz = every "
          f"{1000/p.exchange_hz:.1f} ms = 50 brain steps + 50 physics steps")
    print("  Running the SAME loop with the odour source mirrored left/right.")
    print("  If the fly turns toward the source in BOTH cases, the behaviour is")
    print("  odour-driven. If it turns the same way regardless, it is being")
    print(f"  driven by the {intercept:+.1f} Hz fixed pedestal and my bridge code.")

    SECONDS = 3.0
    episodes = {}
    for name, src in (("srcL", ((22.0, 12.0, 1.5),)),
                      ("srcR", ((22.0, -12.0, 1.5),))):
        print(f"\n  --- odour source at {src[0][:2]} ---")
        episodes[name] = run_closed_loop(
            p, seconds=SECONDS, odor_source=src, label=f"m4_{name}",
            brain=brain, calib=(intercept, slope))
        np.savez_compressed(
            OUT / f"m4_{name}.npz", raster=episodes[name]["raster"],
            **{f"tr_{k}": v for k, v in episodes[name]["trace"].items()})

    rule("ODOUR-DEPENDENCE VERDICT")
    summ = {}
    for name, ep in episodes.items():
        t = ep["trace"]
        net_y = float(t["y"][-1] - t["y"][0])
        summ[name] = {
            "net_y": net_y,
            "mean_bias": float(t["bias"].mean()),
            "mean_contrast": float(t["contrast"].mean()),
            "dist_start": float(t["dist"][0]), "dist_end": float(t["dist"][-1]),
            "got_closer": bool(t["dist"][-1] < t["dist"][0]),
            "dn_l": float(t["dn_l"].mean()), "dn_r": float(t["dn_r"].mean()),
        }
        print(f"  {name}: net_y={net_y:+6.2f} mm  mean_bias={t['bias'].mean():+.3f}  "
              f"mean_contrast={t['contrast'].mean():+.5f}  "
              f"dist {t['dist'][0]:.1f}->{t['dist'][-1]:.1f}  "
              f"DNa02 L/R={t['dn_l'].mean():.1f}/{t['dn_r'].mean():.1f}")

    # Source on the LEFT should give MORE positive net_y than source on the RIGHT.
    dy = summ["srcL"]["net_y"] - summ["srcR"]["net_y"]
    db = summ["srcL"]["mean_bias"] - summ["srcR"]["mean_bias"]
    odour_driven = dy > 0
    print(f"\n  net_y(srcL) - net_y(srcR) = {dy:+.2f} mm   "
          f"(positive => steers toward the odour)")
    print(f"  bias(srcL)  - bias(srcR)  = {db:+.4f}")
    print(f"  ==> {'ODOUR-DEPENDENT steering' if odour_driven else 'NOT odour-dependent -- the source side did not change the turn'}")
    if summ["srcL"]["got_closer"] == summ["srcR"]["got_closer"] is False:
        print("      (neither episode approached the source)")

    res = episodes["srcL"]
    (OUT / "m4_report.json").write_text(json.dumps({
        "episodes": summ,
        "mirror_test": {"delta_net_y_mm": dy, "delta_bias": db,
                        "odour_dependent": bool(odour_driven)},
        "calibration": {"intercept_hz": intercept, "slope_hz_per_contrast": slope,
                        "r2": r2, "contrasts": xs, "asym_hz": ys},
        "params": p.to_dict(),
        "wall_s": res["wall_s"],
        "dist_start": float(res["trace"]["dist"][0]),
        "dist_end": float(res["trace"]["dist"][-1]),
        "got_closer": bool(res["trace"]["dist"][-1] < res["trace"]["dist"][0]),
        "mean_bias": float(res["trace"]["bias"].mean()),
        "video": res["video"],
        "subset_decision": {
            "chose": "full 139k brain",
            "why": ("M1 measured 115 sim-ms per wall-second at batch=1 using "
                    "1.45/11 GB VRAM, so a 2.5 s episode costs ~22 s of brain "
                    "compute. A hop<=3 subset (~42,400 neurons) would buy ~3x "
                    "while discarding ~11% of DNa02 input and requiring its own "
                    "validation against the full model. Not worth it at this "
                    "episode length."),
            "batch": "1 (batching is counterproductive: batch=4 is 8x slower in "
                     "absolute wall clock than batch=1 on this GPU)",
        },
    }, indent=2))
    print(f"\n  wrote {OUT/'m4_report.json'}, {OUT/'m4_intact.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
