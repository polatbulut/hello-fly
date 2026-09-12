"""The bridge: odour -> ORN drive -> connectome -> DN rates -> turning.

EVERY HAND-TUNED CONSTANT IN THIS PROJECT LIVES IN THIS FILE.
That is deliberate. This is the layer where the result stops being derived from
the connectome and starts being a choice I made, so it is all in one place with
its justification (or lack of one) written next to it.

Measured facts this is built on (see out/m4a_lateralisation.json):
  * Unilateral ORN drive DOES lateralise DNa02, ipsiversively, at 3.7 sigma.
  * But DNa02 carries a large FIXED left bias independent of stimulus side:
    left drive -> (L-R) = +63.0 Hz, right drive -> (L-R) = +48.5 Hz. Only the
    14.5 Hz difference between those is odour-dependent; the ~55 Hz pedestal is
    a property of the v783 connectome (a plausible culprit is LAL051, which is
    glutamatergic and has 70 vs 130 left/right synapses onto DNa02).
  * DNa01's separation is 1.5 sigma -- not usable. DNa02 only.
  * FlyGym's measured inter-antennal odour contrast is ~3.8%, not 100%. Scaled
    linearly that is ~0.55 Hz of DN differential, well inside the 3-5 Hz trial
    noise. Amplification is therefore REQUIRED, and it is the single biggest
    piece of non-science here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# FlyGym odour sensor indices, determined experimentally in M2 by moving the
# source to +y and -y and comparing readings. NOT taken from documentation.
LEFT_SENSORS = [0, 2]      # left antenna, left maxillary palp
RIGHT_SENSORS = [1, 3]     # right antenna, right maxillary palp


@dataclass
class BridgeParams:
    """Every free parameter, with an honest provenance tag.

    provenance legend:
      MEASURED  -- fitted from a calibration run in this repo
      DERIVED   -- follows from a measured quantity plus arithmetic
      TUNED     -- I picked it. No literature value exists.
    """

    # --- sensory encoding -------------------------------------------------
    orn_rate_max: float = 200.0
    """TUNED. Peak ORN Poisson rate, Hz. 200 Hz matches the rate upstream uses
    for the validated sugar-GRN experiment, so it is at least a rate the model
    is known to behave sensibly at. Real ORNs saturate ~200-300 Hz, so it is
    biologically plausible, but nothing selects it for THIS experiment."""

    intensity_to_hz: float = 4.0e4
    """TUNED. Scales FlyGym odour intensity (~1e-3 near the source) into Hz.
    Chosen so the fly sits mid-range rather than saturated at spawn. Pure
    engineering; FlyGym's intensity units are arbitrary."""

    contrast_gain: float = 55.0
    """TUNED, AND THIS IS THE LOAD-BEARING ONE.

    Set from the measured contrast distribution rather than picked: during the
    M4 episodes the inter-antennal contrast ranged over roughly +/-0.018, so a
    gain of ~1/0.018 = 55 maps the typical gradient onto the full +/-1 ORN
    contrast range. The first attempt used 12, which only reached 0.22 of the
    available range and therefore produced 8.6 * 0.22 = 1.9 Hz of DN signal
    against ~14 Hz of Poisson noise (SNR 0.13). That was a mis-calibration on
    my part, and it is recorded here rather than quietly overwritten.

    Be clear about what this constant means regardless of its value: the fly is
    NOT resolving a natural odour gradient. The gradient is being amplified ~55x
    before it reaches the ORNs, and the connectome is then asked to transduce
    the amplified version. The transduction is real; the stimulus is not."""

    # --- DN decoding ------------------------------------------------------
    dn_baseline_asym: float = 0.0
    """MEASURED by calibrate(). The fixed (L-R) DNa02 offset under symmetric
    odour. Subtracted so that only the odour-dependent component steers.
    Without this the fly would turn hard left forever, which would LOOK like
    working odour tracking whenever the source happened to be on the left."""

    dn_asym_scale: float = 1.0
    """MEASURED by calibrate(). Hz of (L-R) DNa02 per unit ORN contrast,
    i.e. the slope of the calibration line. Normalises bias to roughly [-1,1]."""

    readout_tau_ms: float = 400.0
    """TUNED, and forced upward by Poisson noise rather than chosen for realism.

    A rate estimated from Poisson spikes over a window T has SD = sqrt(r/T).
    For DNa02 at ~60 Hz:
        T = 50 ms  -> 35 Hz SD
        T = 200 ms -> 17 Hz SD
        T = 300 ms -> 14 Hz SD
        T = 1000 ms ->  8 Hz SD
    The measured calibration slope is only ~8.6 Hz per unit ORN contrast, so the
    signal does not clear its own shot noise until T is of order a second --
    which is longer than the whole behavioural episode. 300 ms is a compromise:
    SNR is still order 1, and the steering is correspondingly sluggish and noisy.

    This is not a tuning detail, it is a genuine limit of the result. The
    connectome's odour-contrast-to-DN gain is weak enough that a single DNa02
    pair cannot carry a clean steering signal on behavioural timescales."""

    # --- motor encoding ---------------------------------------------------
    turn_gain: float = 0.8
    """TUNED. Normalised DN asymmetry -> left/right descending drive.
    NO PUBLISHED VALUE EXISTS for deg/s of turning per spike/s of DN activity.
    The only literature route would be digitising a figure in Rayshubskiy et
    al. This is the largest unconstrained parameter in the motor half."""

    base_drive: float = 1.0
    """TUNED. Forward walking drive when the odour is symmetric."""

    action_clip: tuple = (-0.5, 1.5)
    """FIXED by FlyGym: action_space is Box(-0.5, 1.5, (2,))."""

    # --- loop timing ------------------------------------------------------
    exchange_hz: float = 200.0
    """DERIVED. Brain<->body information exchange rate. Physics and brain each
    advance at their own native 0.1 ms step between exchanges; they are not
    stepped in lockstep. 200 Hz = every 5 ms = 50 native steps of each."""

    def to_dict(self):
        d = asdict(self)
        d["action_clip"] = list(self.action_clip)
        return d


def odour_to_orn_rates(odor_intensity, p: BridgeParams):
    """FlyGym odour observation -> (left_hz, right_hz, contrast, mean_intensity).

    odor_intensity : (n_dims, n_sensors); we use dimension 0.
    """
    inten = np.asarray(odor_intensity)[0]
    left = float(inten[LEFT_SENSORS].mean())
    right = float(inten[RIGHT_SENSORS].mean())
    total = left + right

    # Normalised contrast is scale-free, so it keeps working as absolute
    # intensity rises by orders of magnitude on approach.
    contrast = (left - right) / (total + 1e-12)
    amplified = np.clip(contrast * p.contrast_gain, -1.0, 1.0)

    mean_i = 0.5 * total
    base = float(np.clip(mean_i * p.intensity_to_hz, 0.0, p.orn_rate_max))

    left_hz = float(np.clip(base * (1.0 + amplified), 0.0, p.orn_rate_max))
    right_hz = float(np.clip(base * (1.0 - amplified), 0.0, p.orn_rate_max))
    return left_hz, right_hz, contrast, mean_i


def dn_to_action(dn_left_hz, dn_right_hz, p: BridgeParams):
    """DNa02 left/right firing rates -> FlyGym action, plus the intermediate bias.

    SIGN CHAIN, each link established empirically:
      1. DNa02 is IPSIVERSIVE: higher LEFT DNa02 -> fly turns LEFT.
         (literature, and reproduced in M4a: left ORN drive -> higher left DNa02)
      2. FlyGym action = [left_drive, right_drive]; measured in M2, LOWERING the
         left drive turns the fly LEFT (+122.7 deg vs +10.3 deg baseline).
      3. Therefore higher left DNa02 must LOWER action[0]. Hence the minus sign.
    """
    asym = dn_left_hz - dn_right_hz
    bias = (asym - p.dn_baseline_asym) / (abs(p.dn_asym_scale) + 1e-9)
    bias = float(np.clip(bias, -1.0, 1.0))
    lo, hi = p.action_clip
    action = np.clip(
        [p.base_drive - p.turn_gain * bias, p.base_drive + p.turn_gain * bias],
        lo, hi,
    ).astype(np.float32)
    return action, bias, asym
