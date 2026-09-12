#!/usr/bin/env python3
"""M4a: THE GO/NO-GO. Does unilateral ORN drive actually lateralise the DNs?

This runs before any closed loop, because it decides whether a closed loop can
honestly be called connectome-driven.

Structural prior (measured on v783 during review): DNa01/DNa02 sit exactly 3
synapses downstream of the ORNs via ORN -> ALPN -> LAL -> DN, cholinergic the
whole way, so the sign should be ipsilateral-excitatory. BUT only ~13% of DNa02's
input synapses come from within 2 hops of the ORN population, and no single
presynaptic partner exceeds ~3% of its input. The olfactory drive is weak and
diffuse. A null result here is a real possibility and must be reported as one.

Method: drive left ORNs only, right ORNs only, and both, at several rates.
Measure left and right DNa01/DNa02 spike counts over repeated trials, and
compare the left-right difference against trial-to-trial variability. An
asymmetry that does not clear its own noise is not an asymmetry.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from brain import DT, Brain  # noqa: E402

OUT = REPO / "out"
OUT.mkdir(exist_ok=True)
NEURONS = json.loads((REPO / "neurons.v783.json").read_text())

TRIAL_MS = 1000.0
N_TRIALS = 4
RATES_HZ = [50.0, 200.0]


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def main():
    orn_l = NEURONS["ORN"]["left"]["root_ids"]
    orn_r = NEURONS["ORN"]["right"]["root_ids"]
    dn = {f"{n}_{s}": NEURONS["DN"][n][s]["root_ids"][0]
          for n in ("DNa01", "DNa02") for s in ("left", "right")}

    rule("M4a  does unilateral ORN drive lateralise the DNs?")
    print(f"ORNs: {len(orn_l)} left, {len(orn_r)} right")
    for k, v in dn.items():
        print(f"  {k:12s} {v}")

    # Stimulated neurons get refractory period zeroed (upstream's protocol), so
    # the full ORN population is declared as stim regardless of condition; the
    # condition is set by which entries of `rates` are non-zero.
    brain = Brain(batch=1, stim_ids=orn_l + orn_r, device="cuda")
    dn_cols = torch.as_tensor(brain.idx(list(dn.values())), device="cuda")
    dn_names = list(dn.keys())

    conditions = {"baseline (no drive)": ([], 0.0)}
    for hz in RATES_HZ:
        conditions[f"LEFT ORNs @ {hz:.0f} Hz"] = (orn_l, hz)
        conditions[f"RIGHT ORNs @ {hz:.0f} Hz"] = (orn_r, hz)
        conditions[f"BOTH ORNs @ {hz:.0f} Hz"] = (orn_l + orn_r, hz)

    n_steps = int(TRIAL_MS / DT)
    results = {}
    t_start = time.perf_counter()

    for cname, (ids, hz) in conditions.items():
        rates = brain.zero_rates()
        if ids:
            rates[:, torch.as_tensor(brain.idx(ids), device="cuda")] = hz
        per_trial = []
        for trial in range(N_TRIALS):
            brain.reset(seed=1000 + trial)
            # Accumulate on the GPU: no per-step host sync, which is what made
            # M1's record_all path slow.
            counts = torch.zeros(1, len(dn_cols), device="cuda")
            with torch.no_grad():
                for _ in range(n_steps):
                    s = brain.step(rates)
                    counts += s.index_select(1, dn_cols)
            torch.cuda.synchronize()
            per_trial.append(counts.cpu().numpy().ravel())
        arr = np.array(per_trial)                      # (trials, 4) spikes/s
        results[cname] = arr
        mean, sd = arr.mean(0), arr.std(0, ddof=1)
        print(f"\n{cname}")
        for i, nm in enumerate(dn_names):
            print(f"  {nm:12s} {mean[i]:7.2f} +/- {sd[i]:5.2f} Hz   "
                  f"trials={arr[:, i].astype(int).tolist()}")

    rule("LATERALISATION: left-minus-right, versus trial noise")
    summary = {}
    for cname, arr in results.items():
        row = {}
        for cell in ("DNa01", "DNa02"):
            li, ri = dn_names.index(f"{cell}_left"), dn_names.index(f"{cell}_right")
            diff = arr[:, li] - arr[:, ri]
            m, sd = diff.mean(), diff.std(ddof=1)
            # Effect size against the spread of the difference itself.
            d = m / sd if sd > 1e-9 else (np.inf if abs(m) > 1e-9 else 0.0)
            row[cell] = {"mean_L_minus_R": float(m), "sd": float(sd),
                         "cohens_d": float(d),
                         "L": float(arr[:, li].mean()),
                         "R": float(arr[:, ri].mean())}
            verdict = ("IPSI" if m > 0 else "CONTRA") if abs(d) > 2 else "no effect"
            print(f"  {cname:24s} {cell}: L={arr[:,li].mean():6.2f} "
                  f"R={arr[:,ri].mean():6.2f}  L-R={m:+7.2f} +/-{sd:5.2f}  "
                  f"d={d:+6.2f}  {verdict}")
        summary[cname] = row

    rule("VERDICT")
    ok = []
    for hz in RATES_HZ:
        lc, rc = f"LEFT ORNs @ {hz:.0f} Hz", f"RIGHT ORNs @ {hz:.0f} Hz"
        for cell in ("DNa01", "DNa02"):
            dl = summary[lc][cell]["mean_L_minus_R"]
            dr = summary[rc][cell]["mean_L_minus_R"]
            sep = dl - dr
            noise = max(summary[lc][cell]["sd"], summary[rc][cell]["sd"], 1e-9)
            z = sep / noise
            good = abs(z) > 2
            ok.append(good)
            print(f"  {hz:3.0f} Hz {cell}: (L-R | left drive) = {dl:+6.2f}, "
                  f"(L-R | right drive) = {dr:+6.2f}, separation = {sep:+6.2f} "
                  f"({z:+.1f} sigma)  {'USABLE' if good else 'NOT USABLE'}")
            if good:
                print(f"         sign: driving the LEFT ORNs makes the "
                      f"{'LEFT' if dl > dr else 'RIGHT'} {cell} fire more "
                      f"-> {'IPSIVERSIVE' if dl > dr else 'CONTRAVERSIVE'}")

    usable = any(ok)
    print(f"\n  ==> {'A usable lateralised DN signal EXISTS' if usable else 'NO usable lateralised DN signal'}")
    if not usable:
        print("      The closed loop cannot be honestly called connectome-driven.")
        print("      Report this rather than tuning until something moves.")

    elapsed = time.perf_counter() - t_start
    print(f"\n  wall clock: {elapsed:.1f} s for "
          f"{len(conditions) * N_TRIALS} x {TRIAL_MS:.0f} ms trials")

    (OUT / "m4a_lateralisation.json").write_text(json.dumps({
        "trial_ms": TRIAL_MS, "n_trials": N_TRIALS, "rates_hz": RATES_HZ,
        "dn_names": dn_names,
        "raw_spike_counts": {k: v.tolist() for k, v in results.items()},
        "summary": summary,
        "usable_lateralisation": bool(usable),
        "wall_s": elapsed,
    }, indent=2))
    print(f"  wrote {OUT/'m4a_lateralisation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
