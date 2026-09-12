#!/usr/bin/env python3
"""M1: run the brain alone on GPU, reproduce the sugar->proboscis result, benchmark.

CORRECTNESS CHECK (Shiu et al., Nature 634:210-219, 2024, Fig 1)
Activating 21 labellar sugar-sensing GRNs at 200 Hz should drive proboscis motor
neurons -- MN9 in particular, the rostrum protractor. The paper reports that
unilateral sugar GRN activation drives the CONTRALATERAL MN9 more strongly than
the ipsilateral one.

Quantitative anchor: upstream's own v783 manifest reports 16,353-17,429 spikes
and 380-416 active neurons for a single 1 s trial at 200 Hz across six backends
(the Nature paper's 455 active is on v630, a different materialisation).

BENCHMARK
Reports milliseconds of simulated brain time per second of wall clock, which is
the number M4's loop design depends on.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brain import DT, Brain  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "external/fly-brain/code"))
from benchmark import EXPERIMENTS  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# MN9, the rostrum-protractor proboscis motor neuron. The paper's ID is
# 720575940660219265, annotated in v783 as cell_type CB0701, super_class motor,
# nerve PhN, side RIGHT -- contralateral to the (left) stimulated GRNs.
# The paper's ipsilateral partner 720575940645521262 is DEAD in v783; the only
# other CB0701 is 720575940618238523. Both are checked below.
MN9 = {
    "MN9_contra (paper, v783 side=right)": 720575940660219265,
    "MN9_other  (other CB0701 in v783)": 720575940618238523,
}

ANCHOR = {"spikes": (16353, 17429), "active": (380, 416)}


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    exp = EXPERIMENTS["sugar"]
    grns, hz = exp["neu_exc"], exp["stim_rate"]
    rule(f"M1  experiment={exp['name']}  {len(grns)} GRNs @ {hz} Hz")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    t0 = time.perf_counter()
    brain = Brain(batch=1, stim_ids=grns, device="cuda")
    t_load = time.perf_counter() - t0
    print(f"\nsetup: {t_load:.2f} s   neurons={brain.n_neurons}   "
          f"VRAM={brain.vram_gb():.2f} GB")

    # ---------------------------------------------------------------- run 1 s
    rule("1 s trial, recording all spikes")
    rates = brain.rates_for(grns, hz)
    brain.reset(seed=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = brain.run(rates, duration_ms=1000.0, record_all=True)
    torch.cuda.synchronize()
    t_sim = time.perf_counter() - t0

    sp = res["spikes"]
    n_spikes, n_active = len(sp), sp["neuron_index"].nunique()
    print(f"wall clock     : {t_sim:.2f} s for 1000 ms simulated")
    print(f"total spikes   : {n_spikes}")
    print(f"active neurons : {n_active}")
    print(f"VRAM           : {brain.vram_gb():.2f} GB")

    ok_sp = ANCHOR["spikes"][0] <= n_spikes <= ANCHOR["spikes"][1]
    ok_ac = ANCHOR["active"][0] <= n_active <= ANCHOR["active"][1]
    print(f"\nvs upstream v783 manifest:")
    print(f"  spikes {n_spikes} in {ANCHOR['spikes']}? {'YES' if ok_sp else 'NO'}")
    print(f"  active {n_active} in {ANCHOR['active']}? {'YES' if ok_ac else 'NO'}")
    print("  (stochastic Poisson drive + a different RNG stream than upstream,")
    print("   so landing near the band is the signal, not exact equality)")

    # ------------------------------------------------------- the actual check
    rule("CORRECTNESS: do proboscis motor neurons fire?")
    idx2id = brain.i2flyid
    counts = sp["neuron_index"].value_counts()
    mn_report = {}
    for label, rid in MN9.items():
        if rid not in brain.flyid2i:
            print(f"  {label:38s} root {rid}: NOT IN v783")
            mn_report[label] = None
            continue
        i = brain.flyid2i[rid]
        c = int(counts.get(i, 0))
        rate = c / 1.0
        mn_report[label] = {"root_id": rid, "index": i, "spikes": c, "rate_hz": rate}
        print(f"  {label:38s} root {rid}: {c:4d} spikes  ({rate:.1f} Hz)")

    fired = [k for k, v in mn_report.items() if v and v["spikes"] > 0]
    print(f"\n  -> {len(fired)}/{len([v for v in mn_report.values() if v])} "
          f"MN9 candidates fired")

    rule("20 most active neurons (downstream of the sugar GRNs)")
    stim_set = set(brain.idx(grns).tolist())
    rows = []
    for i, c in counts.head(30).items():
        rows.append({
            "neuron_index": int(i),
            "flywire_id": int(idx2id[int(i)]),
            "spikes": int(c),
            "rate_hz": float(c),
            "is_stimulated_GRN": int(i) in stim_set,
        })
    top = pd.DataFrame(rows)
    print(top.head(20).to_string(index=False))
    n_driven = int((~top["is_stimulated_GRN"]).sum())
    print(f"\n  of the top 30, {n_driven} are downstream (not directly stimulated)")

    # -------------------------------------------------------------- benchmark
    rule("BENCHMARK: simulated ms per second of wall clock")
    print("(no spike recording -- that is I/O, measured separately above)")
    bench = []
    for batch in (1, 4, 8, 16, 32):
        try:
            b = Brain(batch=batch, stim_ids=grns, device="cuda", quiet=True)
            r = b.rates_for(grns, hz)
            b.reset(seed=1)
            b.run(r, duration_ms=20.0)              # warm up kernels
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            b.run(r, duration_ms=200.0)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ms_per_s = 200.0 / dt
            per_step_ms = dt / (200.0 / DT) * 1000
            bench.append({
                "batch": batch,
                "wall_s_for_200ms": round(dt, 3),
                "sim_ms_per_wall_s": round(ms_per_s, 1),
                "realtime_ratio": round(ms_per_s / 1000, 4),
                "ms_per_step": round(per_step_ms, 3),
                "aggregate_sim_ms_per_wall_s": round(ms_per_s * batch, 1),
                "vram_gb": round(b.vram_gb(), 2),
            })
            print(f"  batch={batch:2d}  {dt:6.2f} s / 200 ms sim  ->  "
                  f"{ms_per_s:7.1f} sim-ms per wall-s  "
                  f"({ms_per_s/1000:.4f}x realtime, {per_step_ms:.2f} ms/step, "
                  f"{b.vram_gb():.2f} GB)")
            del b
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  batch={batch:2d}  FAILED: {str(e)[:120]}")
            torch.cuda.empty_cache()
            break

    # ------------------------------------------------------------------ save
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dt_ms": DT,
        "n_neurons": brain.n_neurons,
        "setup_s": round(t_load, 2),
        "trial": {
            "duration_ms": 1000.0, "wall_s": round(t_sim, 2),
            "spikes": int(n_spikes), "active_neurons": int(n_active),
            "anchor": ANCHOR,
            "spikes_within_anchor": bool(ok_sp),
            "active_within_anchor": bool(ok_ac),
        },
        "motor_neurons": mn_report,
        "top_active": rows[:30],
        "benchmark": bench,
    }
    (OUT / "m1_report.json").write_text(json.dumps(report, indent=2))
    sp.to_parquet(OUT / "m1_sugar_spikes.parquet", compression="zstd")
    print(f"\nwrote {OUT/'m1_report.json'} and {OUT/'m1_sugar_spikes.parquet'}")

    rule("M1 SUMMARY")
    print(f"  spikes/active within upstream band : "
          f"{'YES' if (ok_sp and ok_ac) else 'PARTIAL/NO'}")
    print(f"  proboscis MN fired                 : {'YES' if fired else 'NO'}")
    if bench:
        print(f"  throughput (batch=1)               : "
              f"{bench[0]['sim_ms_per_wall_s']:.1f} sim-ms per wall-second "
              f"({bench[0]['realtime_ratio']:.4f}x realtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
