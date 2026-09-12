#!/usr/bin/env python3
"""M5b: the control that CAN fail -- cross the sensory wiring.

WHY M5 ALONE IS NOT ENOUGH
The M5 silencing ablation abolished odour-dependence completely (mirror delta
+14.15 -> +0.00 mm). But the silenced trajectories were identical to two decimal
places across odour conditions, which exposes the problem: the bridge reads ONLY
DNa02, so removing DNa02 necessarily removes every path from odour to motor. The
fly falls back to a fixed open-loop motor program and deterministic physics does
the rest. That test could not have failed.

THIS TEST CAN FAIL
Swap which ORN population receives which antenna's odour -- left antenna drives
the RIGHT ORNs and vice versa. Nothing downstream changes: same connectome, same
DNa02 readout, same bridge constants, same calibration. The fly still walks and
still responds to odour.

If the connectome is genuinely transducing the lateralisation, DNa02's ipsiversive
response must now point the wrong way, and the mirror difference must INVERT:

    intact   mirror delta  = +14.15 mm  (turns toward the odour)
    swapped  mirror delta  <  0         (turns away from the odour)

If instead the swapped delta stays positive, the steering is not being carried by
the connectome's left/right structure at all, and the M5 pass was an artefact of
the bridge's own wiring.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import Brain  # noqa: E402
from bridge import BridgeParams  # noqa: E402
from m4_loop import calibrate, run_closed_loop  # noqa: E402

OUT = REPO / "out"
N = json.loads((REPO / "neurons.v783.json").read_text())

SECONDS = 2.0
SEEDS = [0, 1, 2]
SOURCES = {"srcL": ((22.0, 12.0, 1.5),), "srcR": ((22.0, -12.0, 1.5),)}
INTACT_DELTA = 14.15      # measured in M5, for reference


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    p = BridgeParams()
    orn_all = N["ORN"]["left"]["root_ids"] + N["ORN"]["right"]["root_ids"]
    rule("M5b  crossed sensory wiring")
    brain = Brain(batch=1, stim_ids=orn_all, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(
        [N["DN"]["DNa02"]["left"]["root_ids"][0],
         N["DN"]["DNa02"]["right"]["root_ids"][0]]), device="cuda")

    intercept, slope, r2, _, _ = calibrate(brain, dn_cols, p)
    p.dn_baseline_asym, p.dn_asym_scale = intercept, slope
    print(f"\n  calibration identical to M5: intercept={intercept:+.2f} "
          f"slope={slope:+.2f} R2={r2:.3f}")
    print(f"  {len(SOURCES) * len(SEEDS)} episodes x {SECONDS} s, ORN wiring CROSSED")

    res = {}
    t0 = time.perf_counter()
    done = 0
    for sname, src in SOURCES.items():
        for seed in SEEDS:
            ep = run_closed_loop(
                p, seconds=SECONDS, odor_source=src, label=f"m5b_{sname}_{seed}",
                seed=seed, brain=brain, calib=(intercept, slope),
                render=False, swap_orn=True)
            t = ep["trace"]
            res[f"{sname}|{seed}"] = {
                "source": sname, "seed": seed,
                "net_y": float(t["y"][-1] - t["y"][0]),
                "dist_start": float(t["dist"][0]),
                "dist_end": float(t["dist"][-1]),
                "mean_bias": float(t["bias"].mean()),
                "mean_dn_l": float(t["dn_l"].mean()),
                "mean_dn_r": float(t["dn_r"].mean()),
            }
            done += 1
            el = time.perf_counter() - t0
            print(f"  [{done}/6] swapped|{sname}|{seed}  "
                  f"net_y={res[f'{sname}|{seed}']['net_y']:+7.2f}  "
                  f"dn_L/R={res[f'{sname}|{seed}']['mean_dn_l']:5.1f}/"
                  f"{res[f'{sname}|{seed}']['mean_dn_r']:4.1f}  "
                  f"[{el/60:.1f} min]")

    yL = np.array([v["net_y"] for v in res.values() if v["source"] == "srcL"])
    yR = np.array([v["net_y"] for v in res.values() if v["source"] == "srcR"])
    delta = float(yL.mean() - yR.mean())
    pooled = float(np.sqrt((yL.var(ddof=1) + yR.var(ddof=1)) / 2))
    eff = delta / pooled if pooled > 1e-9 else float("nan")

    rule("RESULT")
    print(f"  net_y srcL : {np.array2string(yL, precision=2)}  mean {yL.mean():+7.2f}")
    print(f"  net_y srcR : {np.array2string(yR, precision=2)}  mean {yR.mean():+7.2f}")
    print(f"  swapped MIRROR DELTA = {delta:+7.2f} mm  "
          f"(pooled SD {pooled:.2f}, effect {eff:+.2f})")
    print(f"  intact  MIRROR DELTA = {INTACT_DELTA:+7.2f} mm  (M5 reference)")

    inverted = delta < 0
    print()
    if inverted:
        verdict = ("PASSES: crossing the sensory wiring INVERTS the steering. "
                   "The connectome's left/right structure is carrying the "
                   "lateralisation -- this control could have failed and did not.")
    elif abs(delta) < 0.4 * abs(INTACT_DELTA):
        verdict = ("PARTIAL: crossing the wiring destroys the steering without "
                   "cleanly inverting it. Consistent with the connectome carrying "
                   "the signal, but the signal is too noisy to show a clean sign "
                   "flip at this episode count.")
    else:
        verdict = ("FAILS: the mirror difference survives crossing the sensory "
                   "wiring with the same sign. The steering is NOT being carried "
                   "by the connectome's lateralisation, and the M5 pass was an "
                   "artefact of the bridge reading only DNa02.")
    print(f"  {verdict}")

    (OUT / "m5b_report.json").write_text(json.dumps({
        "design": {"seconds": SECONDS, "seeds": SEEDS, "swap_orn": True,
                   "calibration": {"intercept": intercept, "slope": slope}},
        "episodes": res,
        "swapped_mirror_delta_mm": delta,
        "intact_mirror_delta_mm": INTACT_DELTA,
        "pooled_sd": pooled, "effect_size": eff,
        "inverted": bool(inverted),
        "verdict": verdict,
        "wall_min": (time.perf_counter() - t0) / 60,
    }, indent=2))
    print(f"\n  wrote {OUT/'m5b_report.json'} "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
