#!/usr/bin/env python3
"""Do the 1,086 duplicate (A,B)+(B,A) pairs actually produce DUPLICATE CONTACTS?

This matters far more than speed. If MuJoCo instantiates a contact for each
registered pair, the fly has been walking with doubled self-collision forces and
every result in this repo was produced under that condition. If MuJoCo dedupes
internally, the duplicates are merely wasted broad-phase work and removing them
would be free and exact.

Distinguish empirically. Do not guess.
"""
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_loop import build_sim  # noqa: E402

sim, cam = build_sim(((22.0, 12.0, 1.5),), output_path=None, seed=0)
sim.reset(seed=0)
mdl = sim.physics.model.ptr
dat = sim.physics.data.ptr

# Which pairs are duplicated, as unordered geom sets?
seen, dup_keys = set(), set()
for i in range(mdl.npair):
    a, b = int(mdl.pair_geom1[i]), int(mdl.pair_geom2[i])
    k = (min(a, b), max(a, b))
    if k in seen:
        dup_keys.add(k)
    seen.add(k)
print(f"npair={mdl.npair:,}  unique={len(seen):,}  duplicated keys={len(dup_keys):,}")

# Walk a while so the legs actually touch each other and the ground.
act = np.array([1.0, 1.0], dtype=np.float32)
for _ in range(3000):
    sim.step(act)

print(f"\nafter 3000 steps: ncon = {dat.ncon}")
key_counts = Counter()
for i in range(dat.ncon):
    c = dat.contact[i]
    a, b = int(c.geom1), int(c.geom2)
    key_counts[(min(a, b), max(a, b))] += 1

multi = {k: v for k, v in key_counts.items() if v > 1}
print(f"  distinct geom pairs in contact : {len(key_counts)}")
print(f"  pairs with MORE THAN ONE contact: {len(multi)}")

# A geom pair can legitimately have several contact points (mesh-mesh, multiccd).
# The diagnostic question is narrower: do pairs that are DUPLICATED IN THE MODEL
# show more contacts than pairs that are not?
dup_in_contact = {k: v for k, v in key_counts.items() if k in dup_keys}
uniq_in_contact = {k: v for k, v in key_counts.items() if k not in dup_keys}
print(f"\n  contacts on DUPLICATED model pairs : "
      f"{sum(dup_in_contact.values())} across {len(dup_in_contact)} pairs "
      f"(mean {np.mean(list(dup_in_contact.values())) if dup_in_contact else 0:.2f})")
print(f"  contacts on UNIQUE model pairs     : "
      f"{sum(uniq_in_contact.values())} across {len(uniq_in_contact)} pairs "
      f"(mean {np.mean(list(uniq_in_contact.values())) if uniq_in_contact else 0:.2f})")

if dup_in_contact:
    print("\n  sample duplicated pairs currently in contact:")
    for k, v in list(dup_in_contact.items())[:6]:
        n1 = mdl.geom(k[0]).name if hasattr(mdl, "geom") else k[0]
        n2 = mdl.geom(k[1]).name if hasattr(mdl, "geom") else k[1]
        print(f"    {n1} <-> {n2}: {v} contact(s)")

print("\ninterpretation:")
if not dup_in_contact:
    print("  No duplicated pair is in contact right now, so this run cannot")
    print("  distinguish the two hypotheses. The duplicates are self-collision")
    print("  pairs between leg segments, which only touch during grooming-like")
    print("  postures -- during normal walking they may simply never fire.")
    print("  => the duplication is most likely WASTED BROAD-PHASE WORK ONLY,")
    print("     but that is an inference, not a measurement.")
elif multi and set(multi) & dup_keys:
    print("  Duplicated pairs DO show multiple contacts. Forces on those pairs")
    print("  are likely doubled. This is a real dynamics bug, and fixing it")
    print("  would CHANGE results -- not a free optimisation.")
else:
    print("  Duplicated pairs appear once each in data.contact, so MuJoCo is")
    print("  deduplicating internally. Removing them is then free and exact.")

sim.close()
