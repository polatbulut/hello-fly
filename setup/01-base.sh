#!/usr/bin/env bash
# Base Ubuntu setup for the hello-fly project. Run as root inside WSL.
set -euo pipefail

USERNAME="polat"

echo "=== whoami: $(whoami) / $(uname -a) ==="

# --- user ------------------------------------------------------------------
if id "$USERNAME" &>/dev/null; then
    echo "user $USERNAME already exists"
else
    adduser --disabled-password --gecos "" "$USERNAME"
    usermod -aG sudo "$USERNAME"
    echo "created user $USERNAME"
fi

# Passwordless sudo. WSL runs no network-facing sshd and the account has no
# password set, so a password prompt would just be an unanswerable blocker for
# automated setup. Set a real password later with `sudo passwd polat` if you want one.
echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$USERNAME
chmod 0440 /etc/sudoers.d/90-$USERNAME

# --- wsl.conf --------------------------------------------------------------
cat > /etc/wsl.conf <<'EOF'
[boot]
systemd=true

[user]
default=polat

[interop]
enabled=true
appendWindowsPath=false

[network]
generateResolvConf=true
EOF
echo "--- /etc/wsl.conf ---"; cat /etc/wsl.conf

# appendWindowsPath=false keeps the enormous Windows PATH out of the Linux
# shell. It stops Windows python.exe/pip.exe from shadowing the conda env,
# which is a classic and very confusing source of "wrong python" bugs.

# --- packages --------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential gcc g++ make cmake pkg-config \
    git git-lfs curl wget ca-certificates gnupg \
    python3-dev \
    libegl1 libgl1 libglx-mesa0 libgl1-mesa-dri libglu1-mesa \
    libosmesa6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    ffmpeg unzip zstd htop jq

git lfs install --system

echo
echo "=== versions ==="
echo "gcc      : $(gcc --version | head -1)"
echo "cmake    : $(cmake --version | head -1)"
echo "git      : $(git --version)"
echo "git-lfs  : $(git lfs version)"
echo "ffmpeg   : $(ffmpeg -version 2>/dev/null | head -1)"

echo
echo "=== GPU passthrough check ==="
ls -la /usr/lib/wsl/lib/ 2>/dev/null | head -20 || echo "NO /usr/lib/wsl/lib -- GPU passthrough missing!"
echo
nvidia-smi || echo "nvidia-smi FAILED inside WSL"

echo
echo "=== 01-base DONE ==="
