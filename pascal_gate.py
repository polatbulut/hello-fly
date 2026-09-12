#!/usr/bin/env python3
"""Prove this PyTorch build actually computes correctly on this GPU.

WHY THIS EXISTS, AND WHY IT IS NOT `assert "sm_61" in get_arch_list()`

Measured on a GTX 1080 Ti, every cu126 wheel from 2.6.0 through 2.14.0 reports:
    ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
sm_61 is absent and always has been -- PyTorch's Linux build matrix goes sm_50,
sm_60, then jumps to sm_70. A literal `sm_61 in arch_list` assertion therefore
can NEVER pass, on any CUDA version, and would reject a working install.

It does not need to pass. CUDA guarantees cubin compatibility from one minor
revision to the next (6.0 -> 6.1), so the sm_60 cubin runs natively on a 6.1
device. Verified independently with nvcc on this GPU: an sm_60-only binary with
PTX stripped -- no JIT escape hatch -- executes correctly on the 1080 Ti.

The failure this is really guarding against is "no kernel image is available for
execution on the device", which happens when there is NO compatible cubin AND no
usable PTX -- what cu128+/CUDA 13 cause by dropping sm_5x/6x outright. Detecting
that requires EXECUTING kernels, not reading a version string.

Exit code 0 = safe to build on. Non-zero = do not proceed.
"""
import sys

import torch

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def main():
    print(f"torch             : {torch.__version__}")
    print(f"built for CUDA    : {torch.version.cuda}")
    archs = torch.cuda.get_arch_list()
    print(f"arch list         : {archs}")

    if not torch.cuda.is_available():
        print("\nFATAL: torch.cuda.is_available() is False -- no GPU visible to torch.")
        return 1

    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device            : {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {major}.{minor}")
    print(f"VRAM              : {props.total_memory / 1024**3:.1f} GiB")
    print(f"SMs               : {props.multi_processor_count}")
    print()

    # ---- static analysis -------------------------------------------------
    # A cubin for sm_{M}{m} runs on a device sm_{M}{n} iff M == M and m <= n.
    cubins = [a for a in archs if a.startswith("sm_")]
    compatible = [a for a in cubins if int(a[3]) == major and int(a[4:]) <= minor]
    ptx = [a for a in archs if a.startswith("compute_")]
    # PTX can only JIT UP to a newer arch, never down to an older one.
    usable_ptx = [a for a in ptx if (int(a[8]), int(a[9:])) <= (major, minor)]

    print("static analysis:")
    print(f"  need a cubin with major=={major} and minor<={minor}")
    print(f"  compatible cubins : {compatible or 'NONE'}")
    print(f"  PTX embedded      : {ptx or 'none'}")
    print(f"  PTX usable here   : {usable_ptx or 'none'}")
    check(
        "a compatible cubin (or usable PTX) is present",
        bool(compatible or usable_ptx),
        f"{compatible} covers sm_{major}{minor}"
        if compatible
        else "only PTX JIT" if usable_ptx else "nothing can run here",
    )
    print()

    # ---- execution: the actual gate ---------------------------------------
    print("execution tests (the real gate):")
    torch.manual_seed(0)
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        check("dense matmul matches CPU",
              torch.allclose((a @ b).cpu(), a.cpu() @ b.cpu(), atol=1e-3))

        # The connectome model is a sparse synaptic matrix times a spike vector.
        # cuSPARSE kernels compile separately from dense BLAS, so a build can
        # pass the dense test and still fail here.
        n, nnz = 4096, 200_000
        idx = torch.randint(0, n, (2, nnz), device="cuda")
        val = torch.randn(nnz, device="cuda")
        sp = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        x = torch.randn(n, 8, device="cuda")
        check("sparse COO mm matches CPU",
              torch.allclose(torch.sparse.mm(sp, x).cpu(),
                             torch.sparse.mm(sp.cpu(), x.cpu()), atol=1e-3),
              f"{nnz} nnz")

        # The ORN drive is Poisson, implemented as torch.bernoulli each timestep.
        frac = torch.bernoulli(torch.full((200_000,), 0.25, device="cuda")).mean().item()
        check("bernoulli sampling", 0.24 < frac < 0.26, f"mean={frac:.4f}, expect ~0.25")

        # The LIF inner loop: threshold, select, accumulate, at full brain width.
        v = torch.randn(138_639, device="cuda")
        spikes = (v > 0).float()
        v = torch.where(spikes.bool(), torch.full_like(v, -52.0), v + 0.1)
        check("LIF-style compare/where/arith at 138,639 wide",
              torch.isfinite(v).all().item())

        # fp64 is materially slower on consumer Pascal (1/32 rate). Not a
        # failure, but worth knowing before anyone reaches for .double().
        d = torch.randn(256, 256, device="cuda", dtype=torch.float64)
        check("float64 works", torch.isfinite(d @ d).all().item(),
              "note: 1/32 rate on GeForce Pascal, avoid in hot loops")

        # Shake out large-allocation pathology before M1 does it for us.
        big = torch.empty(int(1.5e9) // 4, dtype=torch.float32, device="cuda")
        del big
        torch.cuda.empty_cache()
        check("1.5 GB allocation", True)

        torch.cuda.synchronize()
    except RuntimeError as exc:
        msg = str(exc)
        print(f"  FAIL  GPU execution raised: {msg[:300]}")
        if "no kernel image" in msg:
            print("\n  ^^ This IS the Pascal-incompatible-build failure.")
            print("     The wheel has no cubin this device can run and no usable PTX.")
            print("     Reinstall from an older CUDA index (cu126 or earlier).")
        FAILURES.append("execution")

    print()
    if FAILURES:
        print(f"VERDICT: FAIL -- {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("VERDICT: PASS -- this build executes correctly on this GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
