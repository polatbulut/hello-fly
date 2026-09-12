#!/usr/bin/env bash
# Does an sm_60 SASS cubin actually execute on this sm_61 device?
#
# WHY THIS MATTERS
# torch 2.14.0+cu126 ships ['sm_50','sm_60','sm_70',...] -- sm_60 but NOT sm_61.
# The CUDA docs claim binary compatibility "from one minor revision to the next",
# which would imply an sm_60 cubin runs on a 6.1 device. If that holds in practice,
# newer torch is usable on a 1080 Ti despite failing a literal `sm_61 in arch_list`
# check. If it does not hold, the fallback to an sm_61 build is mandatory.
#
# Decisive because we strip PTX: with only a cubin embedded there is no JIT path
# to silently rescue us, so this measures real SASS compatibility.
set -uo pipefail
export PATH=/usr/local/cuda-12.6/bin:$PATH

cat > /tmp/compat.cu <<'EOF'
#include <cstdio>
__global__ void k(int* o) { o[threadIdx.x] = threadIdx.x * 3; }
int main() {
    int *d; if (cudaMalloc(&d, 32*sizeof(int)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 2; }
    k<<<1,32>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("LAUNCH FAILED: %s\n", cudaGetErrorString(e)); return 1; }
    int h[32]; cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    printf("ran ok, out[2]=%d (expect 6)\n", h[2]);
    return h[2] == 6 ? 0 : 1;
}
EOF

run_case () {
    local desc="$1"; shift
    echo "-------------------------------------------------------------"
    echo "CASE: ${desc}"
    echo "  flags: $*"
    if ! nvcc "$@" -o /tmp/compat /tmp/compat.cu 2>/tmp/nvcc.err; then
        echo "  COMPILE FAILED:"; sed 's/^/    /' /tmp/nvcc.err; return
    fi
    [ -s /tmp/nvcc.err ] && { echo "  nvcc warnings:"; sed 's/^/    /' /tmp/nvcc.err; }
    echo "  embedded:"; cuobjdump --list-elf /tmp/compat 2>/dev/null | sed 's/^/    /'
    cuobjdump --list-ptx /tmp/compat 2>/dev/null | sed 's/^/    /'
    echo -n "  RESULT: "; /tmp/compat || echo "  -> exit $?"
}

echo "device: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo

# Control: native sm_61 SASS, no PTX. Must pass.
run_case "sm_61 SASS only (control, must pass)" -gencode arch=compute_61,code=sm_61

# THE question: sm_60 SASS only, no PTX escape hatch.
run_case "sm_60 SASS only, NO PTX (the real question)" -gencode arch=compute_60,code=sm_60

# What torch 2.14 effectively ships for the low end: sm_50 + sm_60 cubins,
# plus PTX only at the TOP of its range (sm_90), which cannot JIT down to 6.1.
run_case "sm_50+sm_60 SASS, PTX only at compute_90 (mimics torch 2.14)" \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_90,code=compute_90

# For contrast: sm_60 PTX present, so JIT can retarget to 6.1.
run_case "compute_60 PTX only (JIT path)" -gencode arch=compute_60,code=compute_60

echo "-------------------------------------------------------------"
echo "=== compat test DONE ==="
