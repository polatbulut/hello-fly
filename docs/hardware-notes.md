# Hardware notes: Pascal, CUDA, and WSL2

Findings measured on the target machine, not read off documentation. Several
contradict what the internet (and the project's own initial assumptions) will
tell you. Keep this file honest; it is the reason the setup scripts look the way
they do.

Target: **GTX 1080 Ti** (GP102, Pascal, compute capability **6.1**, 11 GB),
driver **560.94**, Ubuntu 24.04 under WSL2 on Windows 10 22H2.

---

## 1. The sm_61 myth

**Claim you will hear:** *"Install torch from the cu126 index and verify `sm_61`
is in `torch.cuda.get_arch_list()`."*

**What is actually true:** `sm_61` is **not in any official PyTorch wheel**, and
never has been. Measured here:

| wheel | `get_arch_list()` |
|---|---|
| `torch 2.6.0+cu126` | `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` |
| `torch 2.14.0+cu126` | `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` |

PyTorch's Linux build matrix goes `sm_50`, `sm_60`, then jumps to `sm_70`. So an
`assert "sm_61" in get_arch_list()` gate **can never pass** and would reject a
perfectly working install. Note how plausible the wrong conclusion looks: you see
`sm_60`, think "Pascal is covered", and you happen to be right — but for the
wrong reason, and the same glance would mislead you on a card that genuinely is
unsupported.

### Why it still works

CUDA guarantees cubin compatibility **from one minor revision to the next**: a
cubin built for `sm_60` runs on a `sm_61` device. Verified directly here with
`nvcc`, deliberately stripping PTX so no JIT could rescue the result
(`setup/test-sm60-compat.sh`):

| binary | embedded | runs on 1080 Ti? |
|---|---|---|
| `-gencode arch=compute_61,code=sm_61` | sm_61 cubin | yes (control) |
| `-gencode arch=compute_60,code=sm_60` | sm_60 cubin only, **no PTX** | **yes** |
| sm_50 + sm_60 cubins, PTX only at compute_90 (mimics torch) | as listed | **yes** |
| `-gencode arch=compute_60,code=compute_60` | PTX only | yes (JIT) |

The third row is the important one: it reproduces what a PyTorch wheel actually
contains, and it runs.

### What the real failure looks like

`RuntimeError: no kernel image is available for execution on the device`

This happens when there is **no compatible cubin AND no usable PTX**. PTX can
only JIT *up* to a newer architecture, never *down* — so a wheel carrying only
`compute_90` PTX cannot rescue a 6.1 device. That is exactly what cu128+ and all
CUDA 13.x builds produce for Pascal, because they drop `sm_5x`/`sm_6x` entirely.

### Therefore

The gate is **execution, not inspection** — see `setup/pascal_gate.py`. It runs a
dense matmul, a sparse COO matmul (cuSPARSE compiles separately from dense BLAS,
so it can fail independently), Bernoulli sampling, and a full-brain-width LIF
step, and compares every result against CPU.

Still pin to **cu126**. The reasoning is just different from the folklore: not
"because cu126 has sm_61" (it doesn't), but because cu128+ and CUDA 13.x drop the
`sm_60` cubin that is actually carrying us.

---

## 2. CUDA toolkit version

Installed **12.6** (`nvcc 12.6.85`), from NVIDIA's `wsl-ubuntu` repo.

- Chosen because it **matches driver 560.94**, not because it is the last
  Pascal-capable release. NVIDIA's own guidance names **12.9** as the last
  toolkit supporting offline compilation below CC 7.5.
- **CUDA 13.0 removed** offline compilation for CC < 7.5 outright. Never install
  a 13.x toolkit on this machine; `setup/02-cuda.sh` refuses to fall back to one.
- `nvcc -arch=sm_61` still works and emits at most a deprecation *warning*,
  suppressible with `-Wno-deprecated-gpu-targets`.

Verified: compiled `-arch=sm_61`, executed on device, correct result, and
`cuobjdump --list-elf` confirmed genuine `sm_61` SASS in the binary.

---

## 3. Do not install a Linux GPU driver inside WSL

The single most destructive mistake available here.

Inside WSL the GPU driver is **projected from Windows** into `/usr/lib/wsl/lib`
(`libcuda.so`, `libnvidia-ml.so.1`, `libd3d12.so`, ...). Installing a Linux
driver package overwrites `libcuda.so` there and breaks GPU passthrough until the
distro is reinstalled.

Use NVIDIA's **`wsl-ubuntu`** repo, which ships the toolkit without any driver,
and install **`cuda-toolkit-12-6`** specifically.

**Do not install:** `cuda`, `cuda-12-x`, `cuda-drivers`, `cuda-runtime-12-x`,
`nvidia-driver-*`. All of these pull the driver. (`cuda-runtime-*` is an easy one
to miss — NVIDIA's own meta-package table documents it as installing the driver.)

`setup/02-cuda.sh` asserts after installation that no driver package landed and
aborts if one did. Keep `/usr/lib/wsl/lib` on `LD_LIBRARY_PATH`.

---

## 3b. GeNN and Brian2CUDA on Pascal

Both work. Neither needed an architecture flag.

**PyGeNN 5.4.0** — verified building and executing `sm_61`:

```
backends compiled in : ['cuda', 'single_threaded_cpu']
generated makefile   : -arch sm_61
generated runner     : librunner.1.sm_61.cubin
1000 timesteps       : t=100.0 ms, V finite
```

Notes that cost time:

- **PyGeNN is not on PyPI** (`/pypi/pygenn/json` 404). Install from the GitHub
  archive: `pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip`.
- **`libffi-dev` is a hard build requirement** and the failure is opaque:
  `pkgconfig.pkgconfig.PackageNotFoundError: libffi not found` during
  `get_requires_for_build_wheel`. Also install `libssl-dev` and `swig`.
- **`CUDA_PATH` must be exported BEFORE `pip install`.** `setup.py` gates the
  entire CUDA backend on it, and without it you get a silently **CPU-only**
  build with no error. Check with
  `list(pygenn.genn_model.backend_modules)` — it must contain `'cuda'`.
- **`CUDA_PATH` must also be set at RUNTIME.** GeNN shells out to `nvcc` per
  model and the backend Makefile hard-errors without it.
- **The backend name is lowercase `'cuda'`.** `GeNNModel(..., backend="CUDA")`
  raises `KeyError: 'CUDA'`.
- **No arch flag exists or is needed.** `backend.cc` builds the target from the
  live device: `"sm_" + major + minor`, so a 1080 Ti gets `-arch sm_61`
  automatically. GeNN JIT-compiles per model, so a prebuilt wheel needs no
  Pascal kernels of its own.

**Brian2CUDA 1.0b1** imports and reports `minimal_compute_capability = 5.0`, so
6.1 passes. It hard-pins `brian2==2.10.1` (exact `==`), which is why it lives in
its own conda env rather than alongside torch.

---

## 4. WSL2 configuration

`.wslconfig` (Windows side, `%USERPROFILE%\.wslconfig`):

- `memory=12GB` on a 16 GB host — deliberate, leaves ~4 GB for Windows. Tight.
  Drop to `10GB` if the desktop thrashes during M4 (physics + brain + encode).
- `autoMemoryReclaim=gradual` — without it the VM's working set only ratchets up
  and Windows ends up starved after a long run.
- `sparseVhd` is **not** enabled: WSL 2.7 refuses it and warns of potential data
  corruption. Forcing it needs `--allow-unsafe`; not worth it on a 742 GB volume.

`/etc/wsl.conf` (Linux side):

- `systemd=true`
- `appendWindowsPath=false` — keeps the Windows `PATH` out of the Linux shell so
  Windows `python.exe`/`pip.exe` cannot shadow the conda env. A classic and very
  confusing source of "wrong python" bugs.

**Put the repo in the ext4 filesystem (`~/hello-fly`), not `/mnt/c`.** The 9p
bridge to the Windows filesystem is roughly an order of magnitude slower, and
this project repeatedly reads a ~100 MB parquet and a ~290 MB weight cache.

---

## 5. Riot Vanguard coexistence

WSL2 requires the Hyper-V hypervisor at boot, which kernel anti-cheats may
reject. You do **not** need to uninstall WSL to game:

```powershell
setup/windows/toggle-hypervisor.ps1 -Mode Off    # gaming; reboot
setup/windows/toggle-hypervisor.ps1 -Mode Auto   # WSL2;   reboot
```

`bcdedit /set {current} hypervisorlaunchtype Off|Auto` plus a reboot. WSL and
Ubuntu survive the toggle untouched.

---

## 6. Verified environment

```
python 3.12.14     torch 2.14.0+cu126     numpy 2.5.3
pandas 3.0.5       pyarrow 25.0.1         scipy 1.18.1     matplotlib 3.11.2
CUDA toolkit 12.6.85          driver 560.94          GTX 1080 Ti (6.1, 11 GB, 28 SMs)
```

### Open risks carried into M1

- **pandas 3.0.5 is a major-version jump.** `fly-brain` was written against
  pandas 2.x. Copy-on-write and the new default string dtype are the likely
  friction points. If `read_parquet`/`read_csv` behave oddly in M1, pin
  `pandas<3` before debugging anything else.
- **float64 runs at 1/32 rate** on GeForce Pascal. Keep the LIF loop in fp32.
- **28 SMs.** Small by modern standards; kernel-launch overhead will dominate at
  138,639 neurons, which is the main thing M1's benchmark needs to measure.
