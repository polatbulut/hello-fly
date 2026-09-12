#!/usr/bin/env python3
"""M4 deliverable: side-by-side video of the walking fly and the DN spike raster."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
OUT = REPO / "out"
DT_MS = 0.1
FPS = 30

DN = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R"]
COL = ["#d1495b", "#2e86ab", "#e8a33d", "#6a9955"]


def build(tag, title):
    npz = OUT / f"m4_{tag}.npz"
    mp4 = OUT / f"m4_{tag}_fly.mp4"
    if not npz.exists() or not mp4.exists():
        print(f"  skip {tag}: missing {npz.name if not npz.exists() else mp4.name}")
        return None

    z = np.load(npz)
    raster = z["raster"]                      # (steps, 4) of 0/1
    t_tr = z["tr_t"]
    dn_l, dn_r = z["tr_dn_l"], z["tr_dn_r"]
    bias = z["tr_bias"]
    x, y = z["tr_x"], z["tr_y"]

    frames = imageio.mimread(mp4, memtest=False)
    n_frames = len(frames)
    total_ms = raster.shape[0] * DT_MS
    print(f"  {tag}: {n_frames} frames, {raster.shape[0]} brain steps "
          f"({total_ms:.0f} ms), {int(raster.sum())} DN spikes")

    # Precompute spike times per DN row (ms)
    spike_t = [np.nonzero(raster[:, i])[0] * DT_MS for i in range(raster.shape[1])]

    fh, fw = frames[0].shape[:2]
    panel_w = fw
    dpi = 100
    out_frames = []

    for k, frame in enumerate(frames):
        now_ms = (k / max(n_frames - 1, 1)) * total_ms
        fig = plt.figure(figsize=(panel_w / dpi, fh / dpi), dpi=dpi)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 0.85],
                              hspace=0.55, left=0.13, right=0.97,
                              top=0.90, bottom=0.11)

        # ---- raster -------------------------------------------------------
        ax = fig.add_subplot(gs[0])
        for i, (st, c) in enumerate(zip(spike_t, COL)):
            sel = st[st <= now_ms]
            ax.vlines(sel / 1000.0, i + 0.6, i + 1.4, color=c, lw=0.7)
        ax.set_ylim(0.4, len(DN) + 0.6)
        ax.set_yticks(range(1, len(DN) + 1))
        ax.set_yticklabels(DN, fontsize=7)
        ax.set_xlim(0, total_ms / 1000.0)
        ax.set_xticklabels([])
        ax.set_title(f"{title}   DN spike raster", fontsize=9)
        ax.axvline(now_ms / 1000.0, color="k", lw=0.8, alpha=0.5)
        ax.tick_params(labelsize=7)

        # ---- DN rates -----------------------------------------------------
        ax2 = fig.add_subplot(gs[1])
        m = t_tr <= now_ms / 1000.0
        ax2.plot(t_tr[m], dn_l[m], color=COL[0], lw=1.2, label="DNa02 L")
        ax2.plot(t_tr[m], dn_r[m], color=COL[1], lw=1.2, label="DNa02 R")
        ax2.set_xlim(0, total_ms / 1000.0)
        ax2.set_ylim(0, max(80, float(np.nanmax(dn_l)) * 1.1))
        ax2.set_ylabel("Hz", fontsize=8)
        ax2.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax2.set_xticklabels([])
        ax2.tick_params(labelsize=7)
        ax2.axvline(now_ms / 1000.0, color="k", lw=0.8, alpha=0.5)

        # ---- steering bias ------------------------------------------------
        ax3 = fig.add_subplot(gs[2])
        ax3.axhline(0, color="#999", lw=0.6)
        ax3.plot(t_tr[m], bias[m], color="#444", lw=1.2)
        ax3.set_xlim(0, total_ms / 1000.0)
        ax3.set_ylim(-1.1, 1.1)
        ax3.set_ylabel("bias", fontsize=8)
        ax3.set_xlabel("time (s)", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.axvline(now_ms / 1000.0, color="k", lw=0.8, alpha=0.5)
        ax3.text(0.01, -0.95, "+ = steer left", fontsize=6, color="#666")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        h = min(frame.shape[0], buf.shape[0])
        combo = np.hstack([frame[:h, :, :3], buf[:h]])
        out_frames.append(combo)

    dst = OUT / f"m4_{tag}_sidebyside.mp4"
    imageio.mimwrite(dst, out_frames, fps=FPS, quality=8,
                     macro_block_size=None)
    print(f"  wrote {dst}")
    return dst


def trajectory_figure():
    """Static summary: both mirrored trajectories on one plot."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (tag, src, title) in zip(axes, [
            ("srcL", (22.0, 12.0), "odour source on the fly's LEFT"),
            ("srcR", (22.0, -12.0), "odour source on the fly's RIGHT")]):
        f = OUT / f"m4_{tag}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        x, y, t = z["tr_x"], z["tr_y"], z["tr_t"]
        sc = ax.scatter(x, y, c=t, cmap="viridis", s=6)
        ax.plot(x, y, color="#888", lw=0.6, alpha=0.6)
        ax.scatter([x[0]], [y[0]], marker="o", s=70, facecolor="none",
                   edgecolor="k", label="start", zorder=5)
        ax.scatter([src[0]], [src[1]], marker="*", s=260, color="#d1495b",
                   edgecolor="k", linewidth=0.5, label="odour source", zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.axhline(0, color="#ddd", lw=0.6); ax.axvline(0, color="#ddd", lw=0.6)
        ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=ax, label="t (s)", fraction=0.045)
    fig.suptitle("M4 closed loop: mirrored odour-source trajectories", fontsize=11)
    fig.tight_layout()
    dst = OUT / "m4_trajectories.png"
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    print(f"  wrote {dst}")


if __name__ == "__main__":
    print("building side-by-side videos")
    build("srcL", "odour LEFT")
    build("srcR", "odour RIGHT")
    trajectory_figure()
    sys.exit(0)
