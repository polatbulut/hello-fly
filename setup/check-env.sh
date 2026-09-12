#!/usr/bin/env bash
# Read-only environment verification. Safe to re-run any time.

echo "=== identity ==="
echo "user         : $(whoami)"
echo "pid1         : $(ps -p 1 -o comm=)"
echo "distro       : $(. /etc/os-release; echo "$PRETTY_NAME")"
echo "kernel       : $(uname -r)"

echo
echo "=== memory (expect ~12G total) ==="
free -h | head -2
echo "swap:"
swapon --show 2>/dev/null || echo "  none"
echo "cpus         : $(nproc)"

echo
echo "=== windows PATH leakage (should be empty; appendWindowsPath=false) ==="
echo "$PATH" | tr ':' '\n' | grep -i '/mnt/c' | head -5 || echo "  clean"

echo
echo "=== toolchain ==="
gcc --version | head -1
cmake --version | head -1
git --version
git lfs version
ffmpeg -version 2>/dev/null | head -1

echo
echo "=== GL / EGL libs for MuJoCo ==="
for lib in libEGL.so.1 libGL.so.1 libOSMesa.so.8; do
    p=$(ldconfig -p | grep -m1 "$lib" | awk '{print $NF}')
    echo "  ${lib}: ${p:-MISSING}"
done

echo
echo "=== WSL GPU passthrough (/usr/lib/wsl/lib) ==="
if [ -d /usr/lib/wsl/lib ]; then
    ls /usr/lib/wsl/lib/ | tr '\n' ' '; echo
else
    echo "  MISSING -- GPU passthrough is not present!"
fi

echo
echo "=== nvidia-smi ==="
nvidia-smi || echo "  nvidia-smi FAILED"

echo
echo "=== compute capability ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
    || echo "  query failed"

echo
echo "=== check-env DONE ==="
