#!/usr/bin/env bash
# Verify PyGeNN really code-generates, compiles with nvcc for sm_61, and runs.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate genn
# CUDA_PATH must be present at RUNTIME too: GeNN shells out to nvcc per model
# and its backend Makefile hard-errors without it.
export CUDA_PATH=/usr/local/cuda-12.6
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
cd /tmp && rm -rf genn_check && mkdir genn_check && cd genn_check

python - <<'PYEOF'
import pygenn
from pygenn import GeNNModel
print(f"pygenn {pygenn.__version__}")
from pygenn import genn_model
print(f"backends compiled in: {list(genn_model.backend_modules)}")
assert "cuda" in genn_model.backend_modules, "CUDA backend MISSING (CPU-only build)"

# GeNN derives the nvcc arch from the live device -- there is no flag to set.
# On a 1080 Ti (major=6, minor=1) backend.cc emits '-arch sm_61' automatically.
m = GeNNModel("float", "arch61check", backend="cuda")
m.dt = 0.1
pop = m.add_neuron_population(
    "P", 100, "LIF",
    {"C": 1.0, "TauM": 20.0, "Vrest": -52.0, "Vreset": -52.0,
     "Vthresh": -45.0, "Ioffset": 2.0, "TauRefrac": 2.2},
    {"V": -52.0, "RefracTime": 0.0})
m.build()          # <- code generation + nvcc compile happens here
m.load()
for _ in range(1000):
    m.step_time()
# Read membrane voltage back off the GPU to prove kernels actually executed,
# not merely that the module loaded.
pop.vars["V"].pull_from_device()
v = pop.vars["V"].view
print(f"  built, loaded, stepped 1000 timesteps -> t={m.t:.1f} ms")
print(f"  V after run: min={v.min():.2f} max={v.max():.2f} mV "
      f"(finite={bool(__import__('numpy').isfinite(v).all())})")
print("  VERDICT: PyGeNN CUDA backend WORKS on sm_61")
PYEOF
rc=$?

echo
echo "--- what arch did GeNN actually tell nvcc to target? ---"
grep -rhoE '\-arch[= ]sm_[0-9]+' arch61check_CODE/*.mk arch61check_CODE/Makefile 2>/dev/null | sort -u | sed 's/^/  /' \
  || echo "  (no generated makefile found)"
echo
echo "--- SASS embedded in the generated runner ---"
for so in arch61check_CODE/*.so; do
    [ -e "$so" ] && { echo "  $so:"; cuobjdump --list-elf "$so" 2>/dev/null | head -3 | sed 's/^/    /'; }
done
exit $rc
