---
title: Architecture selection
tags:
  - architectures
  - machine-learning
---

# Architecture Selection

Architecture selection is the process of matching a model family to the input object, task, symmetry, data scale, and compute constraint.

A practical view is:

$$
a^\star
= \arg\min_{a \in \mathcal{A}}
\left[
\widehat{R}(a)
+ \lambda C(a)
+ \gamma S(a)
\right]
$$

$a$ is a candidate architecture, $\widehat{R}(a)$ is validation risk, $C(a)$ is compute or memory cost, $S(a)$ is system or maintenance cost, and $\lambda,\gamma$ encode how much those costs matter.

## Selection Axes

- Input object: vector, sequence, image/grid, graph, set, structure, or multimodal input.
- Symmetry: translation, permutation, rotation, time order, or no clear symmetry.
- Dependency length: local, medium-range, global, or long-context.
- Output type: class, scalar, sequence, structure, ranking, or generated object.
- Data regime: small labeled data, large unlabeled corpus, noisy labels, or distribution shift.
- Compute regime: training memory, inference latency, throughput, storage, and deployment hardware.
- Evaluation risk: leakage, shortcut features, calibration, robustness, and out-of-distribution behavior.

## Rough Mapping

- Grid/image data with local structure: [[concepts/architectures/cnn|CNN]], [[concepts/architectures/u-net|U-Net]], or [[concepts/architectures/vision-transformer|Vision Transformer]].
- Ordered text or biological sequence: [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/rnn|RNN]], or [[concepts/architectures/state-space-model|state-space model]].
- Graph-structured molecules or relational objects: [[concepts/architectures/gnn|GNN]] or [[concepts/architectures/graph-transformer|Graph Transformer]].
- Unordered sets: [[concepts/architectures/deep-sets|Deep Sets]] or [[concepts/architectures/set-transformer|Set Transformer]].
- 3D structures with symmetry constraints: [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]].
- Large conditional compute: [[concepts/architectures/mixture-of-experts|Mixture of experts]].

## Checks

- What is the simplest architecture that should be a baseline?
- What data symmetry does the architecture encode?
- Is the selected architecture solving the task or exploiting the split?
- Does inference cost matter as much as training cost?
- Which failure mode should be tested first?

## Related

- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/index|Evaluation]]
