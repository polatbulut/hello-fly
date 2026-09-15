#!/usr/bin/env python3
"""Plot the living world: odour landscape, trajectory, and the time series."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "out"

TAG = sys.argv[1] if len(sys.argv) > 1 else "intact"
z = np.load(OUT / f"m6_{TAG}.npz")
rep = json.loads((OUT / f"m6_{TAG}_report.json").read_text())

from world import make_world  # noqa: E402
arena = make_world()

x, y, t = z["tr_x"], z["tr_y"], z["tr_t"]
half = max(160.0, float(np.abs(np.concatenate([x, y])).max()) * 1.15)
X, Y, I = arena.field_on_grid(half=half, n=301, z=0.5, dim=0)
_, _, IB = arena.field_on_grid(half=half, n=301, z=0.5, dim=1)

fig = plt.figure(figsize=(14, 6.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.55, wspace=0.22)

# ---- the world ------------------------------------------------------------
ax = fig.add_subplot(gs[:, 0])
# log scale: the plume spans orders of magnitude, linear shows a dot and a void
ax.contourf(X, Y, np.log10(I + 1e-12), levels=24, cmap="YlGn", alpha=0.85)
ax.contour(X, Y, np.log10(IB + 1e-12), levels=8, cmap="Reds", alpha=0.7,
           linewidths=0.9)
sc = ax.scatter(x, y, c=t, cmap="viridis", s=5, zorder=4)
ax.plot(x, y, color="#333", lw=0.5, alpha=0.5, zorder=3)
ax.scatter([x[0]], [y[0]], marker="o", s=90, facecolor="none", edgecolor="k",
           lw=1.6, zorder=6, label="start")
ax.scatter([x[-1]], [y[-1]], marker="X", s=110, color="k", zorder=6, label="end")
for s in arena.sources:
    ax.scatter([s.pos[0]], [s.pos[1]], marker="*", s=340,
               color=s.rgba[:3], edgecolor="k", lw=0.7, zorder=7)
    ax.annotate(s.label, (s.pos[0], s.pos[1]), textcoords="offset points",
                xytext=(9, 9), fontsize=9, weight="bold")
    ax.add_patch(plt.Circle((s.pos[0], s.pos[1]), arena.feed_radius,
                            fill=False, ec="k", ls=":", lw=0.8, zorder=7))
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"odour landscape (log scale) and trajectory — "
             f"{rep['seconds']:.0f} s of fly life"
             + ("  [ODOUR-BLIND NULL]" if rep["blind"] else ""), fontsize=11)
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper left")
fig.colorbar(sc, ax=ax, label="t (s)", fraction=0.04, pad=0.02)
ax.text(0.01, 0.01, f"wind {tuple(arena._wind.round(1))} →   green = attractive, "
        f"red contours = aversive", transform=ax.transAxes, fontsize=7,
        color="#444")

# ---- time series ----------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t, z["tr_odour"], color="#2e7d32", lw=1.1)
ax1.set_ylabel("odour at\nantennae", fontsize=8)
ax1.tick_params(labelsize=7); ax1.set_xticklabels([])
ax1.set_title("what the fly sensed and did", fontsize=10)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t, z["tr_dn_l"], color="#d1495b", lw=1.0, label="DNa02 L")
ax2.plot(t, z["tr_dn_r"], color="#2e86ab", lw=1.0, label="DNa02 R")
ax2.set_ylabel("Hz", fontsize=8)
ax2.legend(fontsize=6, ncol=2, loc="upper right")
ax2.tick_params(labelsize=7); ax2.set_xticklabels([])

ax3 = fig.add_subplot(gs[2, 1])
ax3.axhline(0, color="#bbb", lw=0.6)
ax3.plot(t, z["tr_bias"], color="#444", lw=0.9)
ax3.set_ylim(-1.1, 1.1)
ax3.set_ylabel("steering\nbias", fontsize=8)
ax3.set_xlabel("time (s)", fontsize=8)
ax3.tick_params(labelsize=7)

rt = rep["realtime_factor"]
fig.suptitle(f"hello-fly M6 — persistent multi-source world   "
             f"({rt:.4f}x realtime, {rep['speedup_vs_m4']:.2f}x faster than M4)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
dst = OUT / f"m6_{TAG}_world.png"
fig.savefig(dst, dpi=140)
print(f"wrote {dst}")

# ---- residence summary ----------------------------------------------------
print("\nper-source residence:")
for s in rep["sources"]:
    print(f"  {s['label']:7s} visits={s['visits']:3d}  dwell={s['dwell_s']:6.2f} s  "
          f"consumed={s['consumed']:.3f}")
d = [np.hypot(x - s["pos"][0], y - s["pos"][1]).min() for s in rep["sources"]]
print("\nclosest approach to each source (mm):")
for s, dd in zip(rep["sources"], d):
    print(f"  {s['label']:7s} {dd:7.1f}")
