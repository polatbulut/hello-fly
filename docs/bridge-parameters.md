# Bridge parameters: every place I guessed

This is the honest accounting. The connectome half of this project is inherited
from published work and is not tuned. The bridge is mine, and everything in it
is listed here whether or not it flatters the result.

Provenance tags:

| tag | meaning |
|---|---|
| **MEASURED** | fitted from a calibration run in this repo, reproducible |
| **DERIVED** | follows from a measured quantity plus arithmetic |
| **TUNED** | I picked it. No literature value exists. |

---

## The three that matter

### 1. `contrast_gain = 55` — TUNED — **the load-bearing one**

The fly's two antennae are ~0.3 mm apart. Measured inter-antennal odour contrast
during an episode is about **±0.018** (1.8%). The connectome converts *100%*
contrast into only ~14.5 Hz of DNa02 differential (M4a), so 1.8% natural contrast
yields roughly **0.26 Hz** — against a Poisson noise floor of 8–17 Hz on any DN
rate estimate. Unamplified, the signal is invisible.

`contrast_gain` multiplies the contrast before it reaches the ORNs. At 55, the
typical gradient spans the full ±1 ORN contrast range.

**What this means:** the fly is not resolving a natural odour gradient. The
gradient is amplified ~55× and the connectome transduces the amplified version.
The transduction is real. The stimulus is not.

*History, recorded rather than overwritten:* the first attempt used `12`, which
reached only 0.22 of the available range → 1.9 Hz signal vs ~14 Hz noise
(SNR 0.13), and the mirror test came back **negative**. Raising it to 55 flipped
the mirror test positive. That is a large behavioural conclusion resting on one
constant I chose, and it is the single biggest caveat in this project.

### 2. `turn_gain = 0.8` — TUNED

Maps normalised DN asymmetry to left/right descending drive. **There is no
published constant relating DN firing rate to angular velocity in deg/s per
spike/s.** The only literature route would be digitising Figure 3C of
Rayshubskiy et al. (eLife RP102230). Nothing anchors this value.

### 3. `readout_tau_ms = 400` — TUNED, forced upward by noise

A rate estimated from Poisson spikes over window *T* has SD = √(r/T). For DNa02
at ~60 Hz:

| T | SD |
|---|---|
| 50 ms | 35 Hz |
| 200 ms | 17 Hz |
| 400 ms | 12 Hz |
| 1000 ms | 8 Hz |

The measured calibration slope is only **8.6 Hz per unit ORN contrast**, so the
signal does not clear its own shot noise until *T* approaches a second — longer
than the whole behavioural episode. 400 ms is a compromise; SNR stays of order 1
and the steering is correspondingly sluggish and noisy.

**This is not a tuning detail, it is a limit of the result.** A single DNa02 pair
cannot carry a clean steering signal on behavioural timescales.

---

## Full table

| parameter | value | provenance | notes |
|---|---|---|---|
| `orn_rate_max` | 200 Hz | TUNED | Matches the rate upstream uses for the validated sugar-GRN experiment, and real ORNs do saturate ~200–300 Hz. But nothing selects it for *this* experiment. |
| `intensity_to_hz` | 4.0e4 | TUNED | Scales FlyGym's arbitrary intensity units (~1e-3 near source) into Hz so the fly sits mid-range rather than saturated. Pure engineering. |
| `contrast_gain` | 55 | TUNED | See above. The load-bearing constant. |
| `dn_baseline_asym` | +57.6 Hz | **MEASURED** | Intercept of the calibration line. The connectome's fixed left-bias pedestal. Subtracted so only the odour-dependent component steers. |
| `dn_asym_scale` | +8.6 Hz | **MEASURED** | Slope of the calibration line, Hz of (L−R) per unit ORN contrast. R² = 0.73. |
| `readout_tau_ms` | 400 ms | TUNED | See above. |
| `turn_gain` | 0.8 | TUNED | See above. |
| `base_drive` | 1.0 | TUNED | Forward drive under symmetric odour. |
| `action_clip` | (−0.5, 1.5) | **FIXED** | Imposed by FlyGym: `action_space = Box(-0.5, 1.5, (2,))`. |
| `exchange_hz` | 200 Hz | DERIVED | Brain↔body exchange. 5 ms = 50 native steps each. Chosen so 5 ms stays short vs both the DN membrane time constant (~20 ms) and the stride period (~80 ms). |
| `warmup_ms` | 300 ms | TUNED | Seeds the rate estimator before the fly moves, so the first ~τ is not a transient with the bias pinned at its clip. |
| ORN sensor map | L=[0,2] R=[1,3] | **MEASURED** | Determined in M2 by moving the source to +y and −y and comparing readings. Not taken from documentation. |

---

## Things that are *not* tuned

Worth stating, since the point of this document is the boundary:

- **All LIF parameters** (`tauMem` 20 ms, `vThreshold` −45 mV, `vRest`/`vReset`
  −52 mV, `tRefrac` 2.2 ms, `tauSyn` 5 ms, `tDelay` 1.8 ms, `wScale` 0.275,
  `scalePoisson` 250) are inherited verbatim from upstream / Shiu et al. Not
  touched.
- **The connectome** — 138,639 neurons, 15,091,983 edges — is used as published.
  No edges added, removed, or reweighted.
- **Neuron identity** comes from a checksummed public v783 annotation dump, keyed
  on `cell_type`, never on `hemibrain_type` (which mislabels DNa01), and sides
  taken from the `side` column, never from free-text labels (which are sometimes
  mirrored). See `neurons.v783.json`.
- **The sign chain** is established empirically at every link, not assumed:
  DNa02 ipsiversive (M4a), FlyGym's action semantics (M2), sensor left/right (M2).
- **The CPG** is FlyGym's, unmodified — but see the README: it is hand-written and
  not connectome-derived, which is the project's structural seam.

---

## What would remove these

| parameter | how to get rid of it |
|---|---|
| `contrast_gain` | Nothing available. Would need a connectome whose ORN→DN gain is high enough to transduce a 1.8% gradient, or a plume model with much steeper spatial structure, or many more sensors. |
| `turn_gain` | Digitise Rayshubskiy et al. Fig 3C to get deg/s per spike/s, then fit FlyGym's drive→angular-velocity curve and compose the two. Feasible, not done here. |
| `readout_tau_ms` | Read out a *population* rather than one DNa02 pair. Averaging N independent neurons cuts the noise by √N. The obvious candidates are the other LAL→DN steering neurons; this is the most promising single improvement. |
| `intensity_to_hz` | Calibrate FlyGym's odour units against a measured ORN dose–response curve. |
| `dn_baseline_asym` | Cannot be removed — it is a property of the v783 connectome. It can only be measured and subtracted, which is what we do. |
