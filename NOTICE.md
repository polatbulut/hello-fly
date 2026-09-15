# Scope of the licence, and what to cite

The MIT licence in `LICENSE` covers **the code in this repository only**.

This repository does **not** redistribute any of the following. Each is
downloaded at install or run time and is excluded by `.gitignore`. Each carries
its own terms.

| component | where it comes from | fetched by |
|---|---|---|
| **FlyWire FAFB v783 connectome** — annotations, classification, cell types | `storage.googleapis.com/flywire-data/codex/data/fafb/783/` | `m3_neurons.py` |
| **v783 connectivity + completeness tables** (138,639 neurons, 15,091,983 edges) | [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain) | `setup/06-clone-flybrain.sh` |
| **Leaky integrate-and-fire brain model** | Shiu et al., *Nature* 2024 | via the repo above |
| **FlyGym / NeuroMechFly v2** | [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym), PyPI `flygym==1.2.1` | `setup/07-flygym.sh` |

## Cite the upstream work, not this repository

If you use this, the substantive scientific contributions are upstream. At
minimum:

- **Connectome.** Dorkenwald, S. et al. *Neuronal wiring diagram of an adult
  brain.* Nature **634**, 124–138 (2024). And Schlegel, P. et al. *Whole-brain
  annotation and multi-connectome cell typing of Drosophila.* Nature **634**,
  139–152 (2024).
- **Brain model.** Shiu, P. K., Sterne, G. R., Spiller, N. et al. *A Drosophila
  computational brain model reveals sensorimotor processing.* Nature **634**,
  210–219 (2024). doi:10.1038/s41586-024-07763-9
- **Body model.** Wang-Chen, S. et al. *NeuroMechFly v2: simulating embodied
  sensorimotor control in adult Drosophila.* Nature Methods (2024).
- **Descending neuron function.** Rayshubskiy, A. et al. *Neural control of
  steering in walking Drosophila.* eLife RP102230.

## What is actually original here

The bridge — the sensory encoding, the DN readout, the calibration procedure,
and the closed loop that connects them. It is roughly 1,500 lines, and every
free parameter in it is enumerated with a provenance tag in
[`docs/bridge-parameters.md`](docs/bridge-parameters.md).

Read that file before citing any behavioural result from this repository. The
headline effect is statistically marginal (p = 0.044) and depends on a 55×
amplification of the odour gradient that has no biological justification.
