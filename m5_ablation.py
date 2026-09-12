#!/usr/bin/env python3
"""M5: the honesty check. Silence DNa02 and see whether odour-dependence dies.

WHAT IS ACTUALLY BEING MEASURED, AND WHY IT IS NOT "does the fly still turn"

Silencing DNa02 forces (L-R) to 0, so the bias term (asym - 57.6)/8.6 saturates
at its -1 clip and the fly turns hard and constantly. That is a degenerate
controller, not an absence of turning, and reading it as "the bias disappeared"
or "the bias survived" would both be wrong.

The meaningful quantity is the MIRROR DIFFERENCE:

    delta = net_y(odour on the left) - net_y(odour on the right)

For an odour-driven fly this is large and positive: it goes left when the odour
is left and right when the odour is right. For a fly driven by my bridge code
and the connectome's fixed +57.6 Hz left-bias pedestal, delta is ~0: it does the
same thing regardless of where the odour is. Silencing DNa02 should collapse
delta toward zero. If it does not, the odour-dependence was never flowing
through DNa02.

CONDITIONS
  intact          -- nothing silenced
  DNa02_silenced  -- the experimental ablation
  DNa01_silenced  -- SPECIFICITY CONTROL. M4a showed DNa01's odour-side
                     separation is only 1.5 sigma (not usable), and it is not
                     in the readout path at all, so silencing it should NOT
                     abolish the effect. If it does, something non-specific is
                     happening (e.g. any perturbation destabilises the network)
                     and the DNa02 result would mean nothing.
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

DNA02 = [N["DN"]["DNa02"]["left"]["root_ids"][0],
         N["DN"]["DNa02"]["right"]["root_ids"][0]]
DNA01 = [N["DN"]["DNa01"]["left"]["root_ids"][0],
         N["DN"]["DNa01"]["right"]["root_ids"][0]]
CONDITIONS = {"intact": None, "DNa02_silenced": DNA02, "DNa01_silenced": DNA01}


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    p = BridgeParams()
    orn_all = N["ORN"]["left"]["root_ids"] + N["ORN"]["right"]["root_ids"]
    rule("M5  ablation")
    brain = Brain(batch=1, stim_ids=orn_all, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(DNA02), device="cuda")

    # Calibrate ONCE on the intact network and hold it fixed across conditions.
    # Re-calibrating per condition would silently re-centre the silenced fly and
    # hide exactly the effect we are testing for.
    intercept, slope, r2, _, _ = calibrate(brain, dn_cols, p)
    p.dn_baseline_asym, p.dn_asym_scale = intercept, slope
    print(f"\n  calibration held FIXED across all conditions: "
          f"intercept={intercept:+.2f} slope={slope:+.2f} R2={r2:.3f}")

    n_eps = len(CONDITIONS) * len(SOURCES) * len(SEEDS)
    print(f"  {n_eps} episodes x {SECONDS} s, no rendering")

    results = {}
    t0 = time.perf_counter()
    done = 0
    for cond, sil in CONDITIONS.items():
        for sname, src in SOURCES.items():
            for seed in SEEDS:
                key = f"{cond}|{sname}|{seed}"
                ep = run_closed_loop(
                    p, seconds=SECONDS, odor_source=src, silence_ids=sil,
                    label=f"m5_{cond}_{sname}_{seed}", seed=seed,
                    brain=brain, calib=(intercept, slope), render=False)
                t = ep["trace"]
                results[key] = {
                    "condition": cond, "source": sname, "seed": seed,
                    "net_y": float(t["y"][-1] - t["y"][0]),
                    "net_x": float(t["x"][-1] - t["x"][0]),
                    "dist_start": float(t["dist"][0]),
                    "dist_end": float(t["dist"][-1]),
                    "mean_bias": float(t["bias"].mean()),
                    "mean_dn_l": float(t["dn_l"].mean()),
                    "mean_dn_r": float(t["dn_r"].mean()),
                    "mean_contrast": float(t["contrast"].mean()),
                }
                done += 1
                el = time.perf_counter() - t0
                print(f"  [{done}/{n_eps}] {key:32s} net_y={results[key]['net_y']:+7.2f} "
                      f"dn_L/R={results[key]['mean_dn_l']:5.1f}/"
                      f"{results[key]['mean_dn_r']:4.1f}  "
                      f"bias={results[key]['mean_bias']:+.3f}  "
                      f"[{el/60:.1f} min, eta {el/done*(n_eps-done)/60:.1f} min]")

    # ---------------------------------------------------------------- analyse
    rule("MIRROR DIFFERENCE PER CONDITION")
    summary = {}
    for cond in CONDITIONS:
        yL = np.array([v["net_y"] for v in results.values()
                       if v["condition"] == cond and v["source"] == "srcL"])
        yR = np.array([v["net_y"] for v in results.values()
                       if v["condition"] == cond and v["source"] == "srcR"])
        delta = float(yL.mean() - yR.mean())
        # Pooled SD of the per-episode net_y, as the scale to judge delta against.
        pooled = float(np.sqrt((yL.var(ddof=1) + yR.var(ddof=1)) / 2)) if len(yL) > 1 else float("nan")
        eff = delta / pooled if pooled and np.isfinite(pooled) and pooled > 1e-9 else float("nan")
        dnl = np.mean([v["mean_dn_l"] for v in results.values() if v["condition"] == cond])
        dnr = np.mean([v["mean_dn_r"] for v in results.values() if v["condition"] == cond])
        summary[cond] = {
            "net_y_srcL": yL.tolist(), "net_y_srcR": yR.tolist(),
            "mean_srcL": float(yL.mean()), "mean_srcR": float(yR.mean()),
            "mirror_delta_mm": delta, "pooled_sd": pooled, "effect_size": eff,
            "mean_dn_l": float(dnl), "mean_dn_r": float(dnr),
        }
        print(f"\n  {cond}")
        print(f"    net_y srcL : {np.array2string(yL, precision=2)}  mean {yL.mean():+7.2f}")
        print(f"    net_y srcR : {np.array2string(yR, precision=2)}  mean {yR.mean():+7.2f}")
        print(f"    DNa02 L/R  : {dnl:.1f}/{dnr:.1f} Hz")
        print(f"    MIRROR DELTA = {delta:+7.2f} mm   pooled SD {pooled:5.2f}   "
              f"effect {eff:+.2f}")

    rule("VERDICT")
    d_int = summary["intact"]["mirror_delta_mm"]
    d_ab = summary["DNa02_silenced"]["mirror_delta_mm"]
    d_ctl = summary["DNa01_silenced"]["mirror_delta_mm"]
    retained = d_ab / d_int if abs(d_int) > 1e-9 else float("nan")
    ctl_ret = d_ctl / d_int if abs(d_int) > 1e-9 else float("nan")

    print(f"  intact          mirror delta = {d_int:+7.2f} mm   (baseline)")
    print(f"  DNa02 silenced  mirror delta = {d_ab:+7.2f} mm   "
          f"({retained*100:+.0f}% of intact)")
    print(f"  DNa01 silenced  mirror delta = {d_ctl:+7.2f} mm   "
          f"({ctl_ret*100:+.0f}% of intact, specificity control)")

    abolished = abs(d_ab) < 0.4 * abs(d_int)
    specific = abs(d_ctl) > 0.4 * abs(d_int)
    print()
    if abs(d_int) < 2.0:
        verdict = ("INCONCLUSIVE: the intact mirror difference is itself too "
                   "small to ablate. No claim either way.")
    elif abolished and specific:
        verdict = ("PASSES: silencing DNa02 abolishes the odour-dependence while "
                   "silencing DNa01 does not. The steering is DNa02-mediated.")
    elif abolished and not specific:
        verdict = ("AMBIGUOUS: silencing DNa02 abolishes the effect, but so does "
                   "silencing DNa01. The perturbation is not specific -- any DN "
                   "ablation may be destabilising the readout.")
    else:
        verdict = ("FAILS: the odour-dependence SURVIVES silencing DNa02. The "
                   "behaviour is being produced by the bridge code, not by the "
                   "connectome's DNa02 output.")
    print(f"  {verdict}")

    (OUT / "m5_report.json").write_text(json.dumps({
        "design": {"seconds": SECONDS, "seeds": SEEDS,
                   "conditions": {k: v for k, v in CONDITIONS.items()},
                   "calibration": {"intercept": intercept, "slope": slope, "r2": r2},
                   "metric": "mirror delta = mean net_y(srcL) - mean net_y(srcR)"},
        "episodes": results,
        "summary": summary,
        "verdict": verdict,
        "dna02_retained_fraction": retained,
        "dna01_retained_fraction": ctl_ret,
        "wall_min": (time.perf_counter() - t0) / 60,
    }, indent=2))
    print(f"\n  wrote {OUT/'m5_report.json'}  "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
