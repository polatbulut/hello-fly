# fly-loop

A closed-loop simulated *Drosophila*: a connectome-derived spiking brain driving a
biomechanical body, doing odour-driven walking.

**Brain** — FlyWire v783 leaky integrate-and-fire whole-brain model (Shiu et al., *Nature* 2024),
via the multi-backend GPU fork [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain).
**Body** — [FlyGym / NeuroMechFly v2](https://neuromechfly.org).
**Bridge** — written here. Nobody has released one.

---

## Read this first: what this project is *not*

This is the part that is easy to oversell, so it is stated before anything else.

### The seam: the connectome stops at the neck

**FlyWire covers the brain only.** The leg motor neurons live in the **ventral nerve cord (VNC)**,
which this connectome does not include. Descending neurons (DNs) are therefore the **output
boundary** of the connectome-derived part of this system.

Everything below the DNs — the central pattern generator, the leg coordination rules, the
joint trajectories — is **hand-written** and ships with FlyGym. It is *not* connectome-derived
and it is *not* learned. It is a controller a human wrote, being steered by a signal that
happens to come out of a connectome model.

So the honest description of the pipeline is:

```
odour (FlyGym physics)
  -> Poisson drive to ORNs            [connectome-derived]
  -> ~139k-neuron LIF brain           [connectome-derived]
  -> DNa01 / DNa02 firing rates       [connectome-derived]  <-- the seam is HERE
  -> left/right rate asymmetry
  -> turning gain                     [HAND-TUNED, see below]
  -> FlyGym CPG walking controller    [hand-written, ships with FlyGym]
  -> leg joint torques -> physics
```

Calling this an "end-to-end connectome-driven fly" would be false. Roughly the top half is
connectome-derived; the bottom half is a conventional robotics controller.

### Known scientific weaknesses

These were established during a literature and connectome review before implementation.
They are listed here rather than buried, because they determine how much the final behaviour
is worth believing.

1. **The olfactory drive to DNa02 is weak and diffuse.** Measured on FlyWire v783: only ~13%
   of DNa02's input synapses come from neurons within 2 hops of the ORN population, and no
   single presynaptic partner exceeds ~3.1% of its input. DNa01/DNa02 sit exactly 3 synapses
   downstream of the ORNs, via `ORN -> ALPN (CB0683) -> LAL011 / LAL030b -> DNa02`, and that
   chain is cholinergic (excitatory) end to end. **A null result is entirely plausible** — the
   ablation in M5 exists precisely to detect that.

2. **The turning gain is a free parameter with no literature value.** There is no published
   constant relating DN firing rate to angular velocity in deg/s per spike/s. It must be
   hand-tuned. This is the single largest piece of non-science in the bridge, and it is
   flagged in the source at the point of use.

3. **The model has no valence machinery.** Rayshubskiy et al. report DNa02 activity ipsilateral
   to an attractive stimulus but *contralateral* to an aversive one. That comparison is
   cross-*modality* (fictive odour vs fictive heat), not odour valence — whether an aversive
   odour flips DNa02 lateralisation is untested. Either way, the LIF connectome model cannot
   represent valence, so an odour's sign is imposed by us, not derived.

4. **Absolute firing rates are not comparable to electrophysiology.** The Shiu et al. model
   assumes zero basal firing rate. Only the *relative* left-right difference is meaningful.

5. **Materialisation mismatch risk.** Shiu et al.'s published model was built on FlyWire
   **v630**; the data in `fly-brain` is **v783**. Root IDs are not guaranteed stable across
   materialisations. Neuron identity is therefore resolved from a pinned v783 annotation dump
   with recorded checksums (see M3) rather than from hardcoded IDs.

### What *is* solid

- **The sign convention is confirmed.** DNa02 is **ipsiversive**: higher firing in the *left*
  DNa02 corresponds to a *left* turn. Supported by two papers using different methods
  (electrophysiology and optogenetics) — though note both come from the Wilson lab and share
  co-authors, so they are not fully independent replications.
- **The olfactory -> DNa02 link is published, not invented.** Rayshubskiy et al. drove ORNs
  optogenetically (`Orco-LexA` > CsChrimson), stimulated single antennae, and saw lateralised
  DNa02 responses with steering toward the stimulated antenna.

---

## Hardware

Target machine, and the constraints it imposes:

| | |
|---|---|
| GPU | NVIDIA GTX 1080 Ti — **Pascal, compute capability 6.1 (`sm_61`)**, 11 GB VRAM |
| CPU / RAM | Intel i7-8700K, 16 GB host (12 GB allocated to WSL2) |
| OS | Windows 10 22H2 + WSL2, Ubuntu 24.04 |
| Driver | 560.94 (supports CUDA 12.6) |

### The Pascal trap

**Pascal has been dropped by recent CUDA tooling**, and a plain `pip install torch` can yield
an install that imports fine and then dies at runtime with
`no kernel image is available for execution on the device`. Pin to the **cu126** index and a
**12.x** toolkit.

But the commonly repeated version of this warning is wrong in a way that matters, so:

> **`sm_61` is not in any official PyTorch wheel, and never has been.** Both `torch 2.6.0+cu126`
> and `torch 2.14.0+cu126` report `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`.
> The build matrix goes `sm_50`, `sm_60`, then jumps to `sm_70`. An
> `assert "sm_61" in torch.cuda.get_arch_list()` gate **can never pass** and would reject a
> working install.

It works anyway because CUDA guarantees cubin compatibility from one minor revision to the
next: the **`sm_60` cubin runs natively on a `sm_61` device**. Verified here with `nvcc`, PTX
deliberately stripped so no JIT could mask the result.

So the correct reason to pin cu126 is *not* "because it has `sm_61`" — it doesn't — but because
cu128+ and CUDA 13.x drop the `sm_60` cubin that is actually carrying us, leaving only
`compute_90` PTX, and **PTX only JITs forward to newer architectures, never back to older ones.**

Consequently every CUDA step here is **gated by an executed test, not a version string**:
`setup/pascal_gate.py` runs a dense matmul, a sparse COO matmul, Bernoulli sampling and a
full-brain-width LIF step on the GPU and checks every result against CPU. Details and the
measured compatibility matrix are in [`docs/hardware-notes.md`](docs/hardware-notes.md).

---

## Setup

Scripts are numbered and idempotent. See `docs/hardware-notes.md` for the reasoning behind
each non-obvious choice.

```bash
# from Windows, elevated (enables virtualization, installs WSL2) - then REBOOT
powershell -ExecutionPolicy Bypass -File setup/windows/enable-wsl.ps1

# inside WSL
sudo bash setup/01-base.sh          # user, wsl.conf, toolchain, GL/EGL libs
sudo bash setup/02-cuda.sh          # CUDA 12.6 toolkit + sm_61 compile-and-execute gate
bash setup/03-conda-torch.sh        # miniforge, flyloop env, torch cu126 + sm_61 gate
```

### If Valorant stops working

Enabling WSL2 turns on the Hyper-V hypervisor, which Riot Vanguard may object to.
You do **not** need to uninstall anything:

```powershell
setup/windows/toggle-hypervisor.ps1 -Mode Off    # gaming; reboot
setup/windows/toggle-hypervisor.ps1 -Mode Auto   # WSL2;   reboot
```

---

## Milestones

| | | Status |
|---|---|---|
| **M0** | Environment: WSL2, CUDA 12.6, conda env, torch cu126, GPU execution verified | **done** |
| **M1** | Brain alone on GPU; reproduce sugar GRN -> proboscis MN activation; benchmark | |
| **M2** | Body alone: FlyGym walking, odour observation readable, turning modulated, mp4 | |
| **M3** | Neuron identification: ORN + DNa01/DNa02 root IDs from a pinned v783 dump | |
| **M4** | Close the loop: decoupled rates, side-by-side video + DN spike raster | |
| **M5** | Honesty check: silence DNa02, confirm the turning bias disappears | |

---

## Licence / provenance

Connectome data is FlyWire v783 (Dorkenwald et al., Schlegel et al.). The LIF model follows
Shiu et al., *Nature* 2024. Respect the upstream licences and citation requirements of both.
