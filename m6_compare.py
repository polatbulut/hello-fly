#!/usr/bin/env python3
"""Is the living fly tracking odour, or is it drifting?

Two tests, the second much stronger than the first.

1. BETWEEN-RUN. Intact vs odour-blind null: closest approach to each source,
   odour sensed. n=1 per condition, so this is suggestive at best -- a single
   random walk can wander somewhere denser by luck.

2. WITHIN-RUN (the decisive one). Across all ~12,000 exchanges of a single run,
   does the steering bias correlate with the inter-antennal odour contrast?
   For the intact fly the causal chain contrast -> ORN rates -> DNa02 L/R ->
   bias should make this positive. For the blind fly it CANNOT be anything but
   zero: the contrast is still recorded but never reaches the ORNs. The blind
   run is therefore a true negative control for this statistic, and with
   thousands of samples it has real power where the endpoint comparison does not.

   The bias lags the contrast: the DN readout is an EMA with tau = 400 ms and
   the loop exchanges at 200 Hz, so the peak correlation should sit ~80
   exchanges back. We scan lags and report the profile, not one cherry-picked
   number.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "out"
sys.path.insert(0, str(REPO))


def load(tag):
    f = OUT / f"m6_{tag}.npz"
    if not f.exists():
        return None, None
    return np.load(f), json.loads((OUT / f"m6_{tag}_report.json").read_text())


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def lagged_corr(x, y, lags):
    """corr(x[t-lag], y[t]) for each lag."""
    out = []
    for L in lags:
        a = x[:-L] if L > 0 else x
        b = y[L:] if L > 0 else y
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if a.std() < 1e-12 or b.std() < 1e-12:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return np.array(out)


def perm_test(x, y, lag, n_perm=2000, seed=0):
    """Block permutation: both series are heavily autocorrelated, so shuffling
    samples would wildly overstate significance. Rotate y instead, which keeps
    its autocorrelation intact and destroys only the alignment with x."""
    rng = np.random.default_rng(seed)
    a = x[:-lag] if lag > 0 else x
    b = y[lag:] if lag > 0 else y
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0, 1.0
    obs = float(np.corrcoef(a, b)[0, 1])
    null = np.empty(n_perm)
    for i in range(n_perm):
        k = int(rng.integers(1, n))
        null[i] = np.corrcoef(a, np.roll(b, k))[0, 1]
    p = float((np.abs(null) >= abs(obs)).mean())
    return obs, p


data = {t: load(t) for t in ("intact", "blind")}
have = [t for t, (z, r) in data.items() if z is not None]
print(f"runs available: {have}")
if "intact" not in have:
    sys.exit("no intact run found")

from world import make_world  # noqa: E402
arena = make_world()
SRC = [(s.label, np.array(s.pos[:2])) for s in arena.sources]

rule("1. BETWEEN-RUN  (n=1 each -- suggestive only)")
print(f"{'':10s} {'path mm':>9s} {'net mm':>8s} {'odour mean':>11s} "
      f"{'odour end/start':>16s}   closest approach (mm)")
for tag in have:
    z, rep = data[tag]
    x, y, od = z["tr_x"], z["tr_y"], z["tr_odour"]
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    close = {lab: float(np.hypot(x - p[0], y - p[1]).min()) for lab, p in SRC}
    print(f"  {tag:8s} {path:9.1f} {net:8.1f} {od.mean():11.6f} "
          f"{od[-1]/max(od[0],1e-12):16.1f}   "
          + "  ".join(f"{k}={v:.1f}" for k, v in close.items()))

rule("2. WITHIN-RUN  corr(odour contrast, steering bias) vs lag")
lags = np.array([0, 10, 20, 40, 60, 80, 100, 140, 200, 300])
print(f"  tau_readout = 400 ms at 200 Hz -> expect the peak near lag ~80")
print(f"\n  {'lag(exch)':>10s} " + " ".join(f"{t:>9s}" for t in have))
profiles = {}
for tag in have:
    z, _ = data[tag]
    profiles[tag] = lagged_corr(z["tr_contrast"], z["tr_bias"], lags)
for i, L in enumerate(lags):
    print(f"  {L:10d} " + " ".join(f"{profiles[t][i]:+9.4f}" for t in have))

rule("3. SIGNIFICANCE at the best lag (block permutation, 2000 rotations)")
for tag in have:
    z, _ = data[tag]
    prof = profiles[tag]
    best = int(lags[int(np.argmax(np.abs(prof)))])
    r, p = perm_test(z["tr_contrast"], z["tr_bias"], max(best, 1))
    n = len(z["tr_bias"])
    verdict = ("odour IS steering" if (p < 0.05 and r > 0)
               else "no odour->steering coupling detected")
    print(f"  {tag:8s} n={n:6d}  best lag={best:4d}  r={r:+.4f}  "
          f"p={p:.4f}  -> {verdict}")

if len(have) == 2:
    rule("VERDICT")
    ri = profiles["intact"][int(np.argmax(np.abs(profiles["intact"])))]
    rb = profiles["blind"][int(np.argmax(np.abs(profiles["blind"])))]
    print(f"  intact peak |r| = {abs(ri):.4f}")
    print(f"  blind  peak |r| = {abs(rb):.4f}   (structurally must be ~0)")
    if abs(rb) > 0.1:
        print("\n  WARNING: the blind control shows correlation it cannot legitimately")
        print("  have. Something is leaking odour into the loop -- investigate")
        print("  before believing the intact number.")
    elif abs(ri) > 3 * max(abs(rb), 0.01):
        print("\n  The intact run couples odour contrast to steering; the blind")
        print("  control does not. The coupling is real, which is what the")
        print("  connectome is being asked to provide.")
    else:
        print("\n  The intact coupling is not clearly above the blind control.")
        print("  The approach to FOOD_B should be read as drift, not taxis.")
else:
    print("\n  (run with --blind to get the null control; without it the")
    print("   within-run correlation has no negative reference)")
