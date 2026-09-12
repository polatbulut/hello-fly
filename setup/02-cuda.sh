#!/usr/bin/env bash
# CUDA 12.x toolkit inside WSL2. Run as root.
#
# CRITICAL WSL RULE
# We use NVIDIA's `wsl-ubuntu` repo, NOT the normal ubuntu2404 repo, and we install
# `cuda-toolkit-12-6` and NOT the `cuda` or `cuda-drivers` metapackages.
# Inside WSL the GPU driver is projected from Windows into /usr/lib/wsl/lib.
# Installing a Linux driver package overwrites libcuda.so there and permanently
# breaks GPU passthrough until you reinstall the distro. The wsl-ubuntu repo exists
# precisely because it ships the toolkit without any driver.
set -euo pipefail

echo "=== pre-flight: GPU must already be visible ==="
if ! nvidia-smi > /dev/null 2>&1; then
    echo "FATAL: nvidia-smi does not work. GPU passthrough is broken; fix that before installing CUDA."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

echo
echo "=== adding NVIDIA wsl-ubuntu apt repo ==="
cd /tmp
ARCH=x86_64
BASE="https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/${ARCH}"

wget -q "${BASE}/cuda-wsl-ubuntu.pin"
mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# keyring package name is stable; pull whatever the repo currently publishes
KEYRING=$(wget -qO- "${BASE}/" | grep -o 'cuda-keyring_[0-9.\-]*_all\.deb' | sort -V | tail -1)
echo "keyring package: ${KEYRING}"
wget -q "${BASE}/${KEYRING}"
dpkg -i "${KEYRING}"
rm -f "${KEYRING}"

apt-get update -qq

echo
echo "=== choosing toolkit version ==="
# Driver 560.94 is the CUDA 12.6 driver, so 12.6 is the exact match.
# Fall back to the highest available 12.x if 12-6 is gone. NEVER 13.x: CUDA 13
# dropped sm_61 entirely and would silently give us a Pascal-incompatible nvcc.
AVAILABLE=$(apt-cache search --names-only '^cuda-toolkit-12-[0-9]+$' | awk '{print $1}' | sort -V)
echo "available 12.x toolkits:"; echo "${AVAILABLE}" | sed 's/^/  /'

if echo "${AVAILABLE}" | grep -qx 'cuda-toolkit-12-6'; then
    PKG=cuda-toolkit-12-6
else
    PKG=$(echo "${AVAILABLE}" | tail -1)
    echo "WARNING: cuda-toolkit-12-6 unavailable, falling back to ${PKG}"
fi
[ -n "${PKG}" ] || { echo "FATAL: no CUDA 12.x toolkit available in repo"; exit 1; }
echo "installing: ${PKG}"

DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${PKG}"

# Assert we did not drag in a driver.
echo
echo "=== driver-contamination check ==="
if dpkg -l | grep -E '^ii\s+(cuda-drivers|nvidia-driver-[0-9]+|libnvidia-gl)' ; then
    echo "FATAL: a Linux NVIDIA driver package got installed. This breaks WSL passthrough."
    exit 1
else
    echo "OK: no Linux driver packages installed (correct for WSL)"
fi

CUDA_HOME=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
echo "CUDA_HOME=${CUDA_HOME}"

# --- systemwide env -------------------------------------------------------
cat > /etc/profile.d/cuda.sh <<EOF
export CUDA_HOME=${CUDA_HOME}
export PATH=\${CUDA_HOME}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}
EOF
chmod 0644 /etc/profile.d/cuda.sh
# /usr/lib/wsl/lib must stay on LD_LIBRARY_PATH -- that is where the projected
# libcuda.so lives. Without it, CUDA apps compile but fail to find a driver.

# shellcheck disable=SC1091
source /etc/profile.d/cuda.sh

echo
echo "=== nvcc ==="
nvcc --version

echo
echo "=== THE GATE: can nvcc actually build and run sm_61 code? ==="
cat > /tmp/arch61.cu <<'EOF'
#include <cstdio>
__global__ void k(int* out) { out[threadIdx.x] = threadIdx.x * 2; }
int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("FAIL: no device\n"); return 1; }
    printf("device      : %s\n", p.name);
    printf("compute cap : %d.%d\n", p.major, p.minor);
    int *d; cudaMalloc(&d, 32 * sizeof(int));
    k<<<1, 32>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("FAIL: kernel launch: %s\n", cudaGetErrorString(e)); return 1; }
    int h[32]; cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    printf("kernel out  : %d %d %d (expect 0 2 4)\n", h[0], h[1], h[2]);
    printf("%s\n", (h[1] == 2 && h[2] == 4) ? "PASS: sm_61 kernel executed correctly" : "FAIL: wrong result");
    return (h[1] == 2 && h[2] == 4) ? 0 : 1;
}
EOF

# -arch=sm_61 forces a real Pascal binary: no PTX JIT fallback can mask a failure.
nvcc -arch=sm_61 -o /tmp/arch61 /tmp/arch61.cu
echo "--- compiled for sm_61, running ---"
/tmp/arch61

echo
echo "--- embedded SASS architectures (must list sm_61) ---"
cuobjdump --list-elf /tmp/arch61 || true

echo
echo "=== 02-cuda DONE ==="
