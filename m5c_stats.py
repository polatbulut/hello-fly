#!/usr/bin/env python3
"""M5c: replicate to n=6 and do the statistics properly.

WHY THIS EXISTS
M5 reported a +14.15 mm intact mirror difference and called it a pass. With
n=3 per cell that is t = 1.98, df = 4, p ~ 0.12 -- NOT significant. The effect
size of 1.61 looks reassuring and is not, because effect size says nothing about
how many samples produced it. Publishing the pass without the p-value would have
been exactly the kind of overselling this project is supposed to avoid.

This adds seeds 3,4,5 to both the intact and crossed-wiring conditions, pools
with the existing seeds 0,1,2, and runs a Welch t-test on each.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import Brain  # noqa: E402
from bridge import BridgeParams  # noqa: E402
from m4_loop import calibrate, run_closed_loop  # noqa: E402

OUT = REPO / "out"
N = json.loads((REPO / "neurons.v783.json").read_text())

SECONDS = 2.0
NEW_SEEDS = [3, 4, 5]
SOURCES = {"srcL": ((22.0, 12.0, 1.5),), "srcR": ((22.0, -12.0, 1.5),)}
CONDS = {"intact": False, "swapped": True}       # name -> swap_orn


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def load_existing():
    """Pull seeds 0,1,2 out of the M5 / M5b reports."""
    got = {"intact": {"srcL": [], "srcR": []},
           "swapped": {"srcL": [], "srcR": []}}
    f5 = OUT / "m5_report.json"
    if f5.exists():
        for v in json.loads(f5.read_text())["episodes"].values():
            if v["condition"] == "intact":
                got["intact"][v["source"]].append(v["net_y"])
    f5b = OUT / "m5b_report.json"
    if f5b.exists():
        for v in json.loads(f5b.read_text())["episodes"].values():
            got["swapped"][v["source"]].append(v["net_y"])
    return got


def main():
    p = BridgeParams()
    orn_all = N["ORN"]["left"]["root_ids"] + N["ORN"]["right"]["root_ids"]
    rule("M5c  replication to n=6 + statistics")
    brain = Brain(batch=1, stim_ids=orn_all, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(
        [N["DN"]["DNa02"]["left"]["root_ids"][0],
         N["DN"]["DNa02"]["right"]["root_ids"][0]]), device="cuda")
    intercept, slope, r2, _, _ = calibrate(brain, dn_cols, p)
    p.dn_baseline_asym, p.dn_asym_scale = intercept, slope

    data = load_existing()
    for c in data:
        for s in data[c]:
            print(f"  existing {c:8s} {s}: n={len(data[c][s])}")

    n_new = len(CONDS) * len(SOURCES) * len(NEW_SEEDS)
    print(f"\n  running {n_new} new episodes (seeds {NEW_SEEDS})")
    t0 = time.perf_counter()
    done = 0
    for cname, swap in CONDS.items():
        for sname, src in SOURCES.items():
            for seed in NEW_SEEDS:
                ep = run_closed_loop(
                    p, seconds=SECONDS, odor_source=src,
                    label=f"m5c_{cname}_{sname}_{seed}", seed=seed,
                    brain=brain, calib=(intercept, slope), render=False,
                    swap_orn=swap)
                t = ep["trace"]
                ny = float(t["y"][-1] - t["y"][0])
                data[cname][sname].append(ny)
                done += 1
                el = time.perf_counter() - t0
                print(f"  [{done}/{n_new}] {cname}|{sname}|{seed}  net_y={ny:+7.2f}  "
                      f"[{el/60:.1f} min, eta {el/done*(n_new-done)/60:.1f} min]")

    rule("POOLED STATISTICS")
    summary = {}
    for cname in CONDS:
        yL = np.array(data[cname]["srcL"])
        yR = np.array(data[cname]["srcR"])
        delta = float(yL.mean() - yR.mean())
        t_stat, p_val = stats.ttest_ind(yL, yR, equal_var=False)
        pooled = float(np.sqrt((yL.var(ddof=1) + yR.var(ddof=1)) / 2))
        # 95% CI on the difference of means (Welch)
        se = float(np.sqrt(yL.var(ddof=1) / len(yL) + yR.var(ddof=1) / len(yR)))
        df = (yL.var(ddof=1)/len(yL) + yR.var(ddof=1)/len(yR))**2 / (
            (yL.var(ddof=1)/len(yL))**2/(len(yL)-1)
            + (yR.var(ddof=1)/len(yR))**2/(len(yR)-1))
        tcrit = stats.t.ppf(0.975, df)
        ci = (delta - tcrit * se, delta + tcrit * se)
        summary[cname] = {
            "n_per_cell": [len(yL), len(yR)],
            "net_y_srcL": yL.tolist(), "net_y_srcR": yR.tolist(),
            "mean_srcL": float(yL.mean()), "mean_srcR": float(yR.mean()),
            "mirror_delta_mm": delta, "pooled_sd": pooled,
            "effect_size": delta / pooled if pooled > 1e-9 else float("nan"),
            "welch_t": float(t_stat), "p_value": float(p_val),
            "df": float(df), "ci95": [float(ci[0]), float(ci[1])],
        }
        sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"\n  {cname}  (n={len(yL)} vs {len(yR)})")
        print(f"    srcL : {np.array2string(yL, precision=2)}  mean {yL.mean():+7.2f}")
        print(f"    srcR : {np.array2string(yR, precision=2)}  mean {yR.mean():+7.2f}")
        print(f"    mirror delta = {delta:+7.2f} mm   95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        print(f"    Welch t = {t_stat:+.3f}, df = {df:.1f}, p = {p_val:.4f}  -> {sig}")

    rule("VERDICT")
    di, pi = summary["intact"]["mirror_delta_mm"], summary["intact"]["p_value"]
    ds, ps = summary["swapped"]["mirror_delta_mm"], summary["swapped"]["p_value"]
    print(f"  intact   delta = {di:+7.2f} mm, p = {pi:.4f}")
    print(f"  swapped  delta = {ds:+7.2f} mm, p = {ps:.4f}")
    print(f"  reduction on crossing the wiring: "
          f"{(1 - abs(ds)/abs(di))*100:.0f}%" if abs(di) > 1e-9 else "")
    print()
    if pi < 0.05 and ps >= 0.05:
        v = ("The intact odour-dependence is statistically significant and it "
             "does NOT survive crossing the sensory wiring. Taken with M5 "
             "(silencing DNa02 abolishes it), the steering is carried by the "
             "connectome's DNa02 lateralisation.")
    elif pi < 0.05 and ps < 0.05 and np.sign(ds) == np.sign(di):
        v = ("The effect is significant BUT survives crossing the wiring with "
             "the same sign. It is not being carried by the connectome's "
             "left/right structure. Do not claim connectome-driven steering.")
    elif pi >= 0.05:
        v = (f"UNDERPOWERED: the intact effect is {di:+.2f} mm but p = {pi:.3f}. "
             "At this episode count and noise level the result is suggestive, "
             "NOT established. More seeds or longer episodes are required "
             "before claiming odour-driven steering.")
    else:
        v = "Mixed; see the numbers above."
    print(f"  {v}")

    (OUT / "m5c_report.json").write_text(json.dumps({
        "design": {"seconds": SECONDS, "new_seeds": NEW_SEEDS,
                   "pooled_with": ["m5_report.json", "m5b_report.json"],
                   "test": "Welch two-sample t-test on net_y, srcL vs srcR"},
        "summary": summary, "verdict": v,
        "wall_min": (time.perf_counter() - t0) / 60,
    }, indent=2))
    print(f"\n  wrote {OUT/'m5c_report.json'} "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
