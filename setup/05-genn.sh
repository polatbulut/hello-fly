#!/usr/bin/env bash
# PyGeNN + Brian2CUDA, each in its OWN conda env. Run as polat.
#
# Separate envs deliberately: brian2cuda hard-pins brian2==2.10.1 (exact ==),
# which is a live threat to the `flyloop` env that just passed the Pascal gate.
# GeNN needs no arch flag -- it JIT-compiles per model and derives -arch sm_61
# from the live device.
set -uo pipefail     # NOT -e: we want to report partial success

source "$HOME/miniforge3/etc/profile.d/conda.sh"

export CUDA_PATH=/usr/local/cuda-12.6
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
echo "CUDA_PATH=${CUDA_PATH}  (must be set BEFORE pip install pygenn)"
nvcc --version | tail -1

GENN_OK=no
B2C_OK=no

# ------------------------------------------------------------------ PyGeNN
echo
echo "=================================================================="
echo "PyGeNN"
echo "=================================================================="
conda env list | awk '{print $1}' | grep -qx genn || conda create -y -q -n genn python=3.12 pip
conda activate genn

# PyGeNN is NOT on PyPI (both /pypi/pygenn/json and /simple/pygenn/ return 404).
# Official install is a source build from the GitHub archive. setup.py gates the
# entire CUDA backend on os.environ["CUDA_PATH"] existing -- miss it and you get
# a silently CPU-only build with no error, which is why CUDA_PATH is exported above.
python -m pip install -q numpy psutil setuptools wheel
for TAG in 5.4.0 5.3.0; do
    echo "--- trying GeNN ${TAG} ---"
    if python -m pip install "https://github.com/genn-team/genn/archive/refs/tags/${TAG}.zip" 2>&1 | tail -20; then
        if python -c "import pygenn" 2>/dev/null; then
            echo "pygenn imported (tag ${TAG})"
            GENN_TAG="${TAG}"
            break
        fi
    fi
    echo "GeNN ${TAG} failed"
done

if python -c "import pygenn" 2>/dev/null; then
    echo
    echo "--- does it have the CUDA backend, or did it silently go CPU-only? ---"
    python - <<'PYEOF'
import sys
try:
    import pygenn
    print("  pygenn", getattr(pygenn, "__version__", "?"))
    from pygenn import GeNNModel
    m = GeNNModel("float", "archcheck", backend="cuda")
    print("  CUDA backend constructed OK")
    # Build a trivial model so GeNN actually shells out to nvcc. This is the
    # real test: it proves codegen + compile + load works for sm_61.
    pop = m.add_neuron_population("P", 10, "LIF",
        {"C":1.0,"TauM":20.0,"Vrest":-65.0,"Vreset":-65.0,"Vthresh":-50.0,"Ioffset":0.0,"TauRefrac":2.0},
        {"V":-65.0,"RefracTime":0.0})
    m.dt = 0.1
    m.build()
    m.load()
    for _ in range(10):
        m.step_time()
    print(f"  built, loaded and stepped 10 timesteps -> t={m.t:.1f} ms")
    print("  VERDICT: GeNN CUDA backend WORKS on sm_61")
except Exception as e:
    print(f"  VERDICT: FAIL -- {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
    [ $? -eq 0 ] && GENN_OK=yes
fi
conda deactivate

# --------------------------------------------------------------- Brian2CUDA
echo
echo "=================================================================="
echo "Brian2CUDA  (separate env: hard-pins brian2==2.10.1)"
echo "=================================================================="
conda env list | awk '{print $1}' | grep -qx b2cuda || conda create -y -q -n b2cuda python=3.12 pip
conda activate b2cuda
python -m pip install -q brian2cuda 2>&1 | tail -10

python - <<'PYEOF'
import sys
try:
    import brian2, brian2cuda
    print(f"  brian2     : {brian2.__version__}")
    print(f"  brian2cuda : {brian2cuda.__version__}")
    from brian2.devices.device import get_device
    # Brian2CUDA derives -arch=sm_XY by stripping the dot off compute_capability.
    # minimal_compute_capability is 5.0 in code, so 6.1 passes.
    import brian2cuda.device as d
    print("  minimal_compute_capability:", d.CUDAStandaloneDevice().minimal_compute_capability)
    print("  VERDICT: brian2cuda importable")
except Exception as e:
    print(f"  VERDICT: FAIL -- {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
[ $? -eq 0 ] && B2C_OK=yes
conda deactivate

echo
echo "=================================================================="
echo "SUMMARY   PyGeNN=${GENN_OK}   Brian2CUDA=${B2C_OK}"
echo "=================================================================="
