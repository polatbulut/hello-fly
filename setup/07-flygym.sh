#!/usr/bin/env bash
# M2: install FlyGym 1.2.1 (NOT 2.x) and verify torch still works afterwards.
#
# WHY 1.2.1 AND NOT 2.1.0
# FlyGym 2.0 (April 2026) was a full rewrite that removed BOTH properties this
# project depends on:
#   - olfaction: there is no olfaction package, no OdorArena, no odor sensors.
#     src/flygym/ has vision/ but nothing for smell. Verified structurally.
#   - Gymnasium compliance: v2.0.0 release notes say "we therefore decided to
#     move away from Gymnasium compliance". Simulation.step() -> None, no obs dict.
# 1.2.1 (July 2025) is the last release with OdorArena + obs["odor_intensity"]
# + HybridTurningNMF, and it still supports Python 3.12 (requires >=3.10,<3.13).
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate flyloop

echo "=== installing flygym==1.2.1 ==="
python -m pip install "flygym==1.2.1"

echo
echo "=== did installing flygym break torch? (re-run the Pascal gate) ==="
python "$HOME/fly-loop/setup/pascal_gate.py" || echo "!!! GATE REGRESSED !!!"

echo
echo "=== versions ==="
python - <<'PYEOF'
import flygym, mujoco, torch, numpy
print(f"  flygym : {flygym.__version__}")
print(f"  mujoco : {mujoco.__version__}")
print(f"  torch  : {torch.__version__}")
print(f"  numpy  : {numpy.__version__}")
PYEOF
