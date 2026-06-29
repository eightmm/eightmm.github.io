---
title: Architecture-objective Fit
tags:
  - architectures
  - objective
---

# Architecture-objective Fit

Architecture-objective fit asks whether a reported improvement comes from the model structure, the training objective, the data representation, or the evaluation protocol. It is a guardrail against attributing every gain to an architecture name.

$$
\text{performance}
\approx
g(\mathcal{F}_{\mathrm{arch}}, \mathcal{L}_{\mathrm{objective}}, r(x), D, E)
$$

where $\mathcal{F}_{\mathrm{arch}}$ is the function class, $\mathcal{L}_{\mathrm{objective}}$ is the training signal, $r(x)$ is the representation, $D$ is the data distribution, and $E$ is the evaluation protocol.

## Fit Matrix

| Architecture bias | Objective that often fits | Example risk |
| --- | --- | --- |
| local sharing | reconstruction, dense prediction, denoising | better augmentation may be credited to CNN/U-Net |
| sequence interaction | language modeling, masked modeling, autoregression | tokenization or context length dominates |
| graph message passing | node/edge/graph prediction, contrastive graph pairs | graph construction leaks target relation |
| equivariance | coordinate prediction, forces, pose refinement | invariant target does not need equivariant output |
| state-space recurrence | long sequence prediction or filtering | objective may not require long memory |
| sparse routing | multi-domain or conditional compute | routing balance and serving cost ignored |

## Attribution Checks

- Are data, objective, compute, and evaluation held constant when claiming architecture superiority?
- Does the architecture bias match the input symmetry, locality, ordering, or geometry?
- Does the output require invariant, equivariant, sequential, graph-level, or set-level behavior?
- Is graph construction, tokenization, pocket extraction, or representation preprocessing part of the method?
- Are ablations strong enough to separate architecture from objective and data scale?

## Related

- [[ai/architectures|Architectures]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/learning/objective-taxonomy|Objective taxonomy]]
