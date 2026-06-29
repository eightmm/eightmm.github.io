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

## Claim Decomposition

When a paper says "architecture X improves performance", split the claim before accepting it:

| Factor | Question | Evidence |
| --- | --- | --- |
| Architecture | What function class or inductive bias changed? | matched objective/data ablation |
| Objective | Did the loss, pretraining signal, or supervision change? | same architecture with different objective |
| Representation | Did tokenization, graph construction, conformer generation, or feature extraction change? | frozen representation or preprocessing ablation |
| Data | Did scale, filtering, split, or label semantics change? | fixed data protocol and leakage check |
| Compute | Did parameter count, training steps, context length, or search budget change? | compute-matched comparison |
| Evaluation | Did metric, threshold, benchmark, or selection rule change? | fixed final test boundary |

The useful comparison is rarely:

$$
\text{architecture A} \;>\; \text{architecture B}
$$

It is usually:

$$
(\mathcal{F}_A,\mathcal{L},r,D,E,C)
\quad\text{vs.}\quad
(\mathcal{F}_B,\mathcal{L},r,D,E,C)
$$

where only the architecture family $\mathcal{F}$ should change if the claim is truly architecture-specific.

## Attribution Checks

- Are data, objective, compute, and evaluation held constant when claiming architecture superiority?
- Does the architecture bias match the input symmetry, locality, ordering, or geometry?
- Does the output require invariant, equivariant, sequential, graph-level, or set-level behavior?
- Is graph construction, tokenization, pocket extraction, or representation preprocessing part of the method?
- Are ablations strong enough to separate architecture from objective and data scale?
- Is the comparison fair under both quality and system constraints such as latency, memory, and throughput?

## Related

- [[ai/architectures|Architectures]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/learning/objective-taxonomy|Objective taxonomy]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
