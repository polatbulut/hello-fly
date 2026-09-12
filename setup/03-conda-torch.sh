#!/usr/bin/env bash
# Miniforge + the flyloop conda env + PyTorch verified to actually run on Pascal.
# Run as the normal user (NOT root). Idempotent.
#
# The gate is setup/pascal_gate.py, and it tests EXECUTION rather than inspecting
# get_arch_list(). See docs/hardware-notes.md section 1 for why: sm_61 is absent
# from every official PyTorch wheel, so the obvious assertion can never pass, and
# the sm_60 cubin is what actually carries a 1080 Ti.
set -euo pipefail

ENV_NAME=flyloop
PY_VER=3.12                       # FlyGym 2.1 requires >=3.12,<3.15
CU_INDEX="https://download.pytorch.org/whl/cu126"
MF_DIR="$HOME/miniforge3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- miniforge
# Miniforge, not Anaconda: conda-forge only, no commercial ToS, no defaults
# channel to conflict with conda-forge builds.
if [ ! -d "$MF_DIR" ]; then
    echo "=== installing miniforge ==="
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$MF_DIR"
    rm -f /tmp/miniforge.sh
else
    echo "=== miniforge already present ==="
fi

# shellcheck disable=SC1091
source "$MF_DIR/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
echo "conda: $(conda --version)"

# ---------------------------------------------------------------- env
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "=== creating env ${ENV_NAME} (python ${PY_VER}) ==="
    conda create -y -n "$ENV_NAME" "python=${PY_VER}" pip
else
    echo "=== env ${ENV_NAME} already exists ==="
fi
conda activate "$ENV_NAME"
echo "python: $(python -V) -> $(which python)"
python -m pip install --upgrade pip -q

# ---------------------------------------------------------------- torch
echo
echo "=== torch from the cu126 index ==="
# --upgrade is load-bearing: without it, an older torch already in the env is
# treated as already-satisfied and pip silently no-ops.
python -m pip install --upgrade torch --index-url "${CU_INDEX}"

echo
echo "=== scientific stack (PyPI, not the torch index) ==="
python -m pip install -q \
    "numpy>=2.0,<3" \
    "pandas>=2.2" \
    "pyarrow>=17" \
    "scipy>=1.15" \
    "matplotlib>=3.10" \
    tqdm

# ---------------------------------------------------------------- the gate
echo
echo "=================================================================="
echo "PASCAL GATE"
echo "=================================================================="
python "${HERE}/pascal_gate.py"

echo
echo "=== versions ==="
python - <<'PYEOF'
import sys, torch, numpy, pandas, pyarrow, scipy, matplotlib
print(f"  python    : {sys.version.split()[0]}")
print(f"  torch     : {torch.__version__}")
print(f"  numpy     : {numpy.__version__}")
print(f"  pandas    : {pandas.__version__}")
print(f"  pyarrow   : {pyarrow.__version__}")
print(f"  scipy     : {scipy.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
PYEOF

echo
echo "=== 03-conda-torch DONE ==="
