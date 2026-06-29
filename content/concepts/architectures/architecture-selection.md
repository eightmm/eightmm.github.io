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

## Decision Table

| First question | If yes | Start with |
| --- | --- | --- |
| Is the input a fixed feature vector? | feature order has no spatial or graph meaning | [[concepts/architectures/mlp|MLP]], linear baseline, tree baseline |
| Is the input a dense grid? | neighbors have stable spatial meaning | [[concepts/architectures/cnn|CNN]], [[concepts/architectures/u-net|U-Net]], [[concepts/architectures/vision-transformer|Vision Transformer]] |
| Is the input an ordered sequence? | token order and context matter | [[concepts/architectures/rnn|RNN]], [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/state-space-model|State-space model]] |
| Is the input an unordered set? | permutation should not change output | [[concepts/architectures/deep-sets|Deep Sets]], [[concepts/architectures/set-transformer|Set Transformer]] |
| Is the input a graph? | edges define local relations | [[concepts/architectures/gnn|GNN]], [[concepts/architectures/graph-transformer|Graph Transformer]] |
| Are 3D coordinates essential? | rotations/translations should be handled explicitly | [[concepts/geometric-deep-learning/equivariance|Equivariance]], [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]] |
| Is capacity large but active compute constrained? | not every input needs all parameters | [[concepts/architectures/mixture-of-experts|Mixture of experts]] |
| Are multiple modalities fused? | timing and alignment between modalities matter | [[concepts/architectures/cross-attention|Cross-attention]], [[concepts/architectures/perceiver|Perceiver]] |

## Rough Mapping

- Grid/image data with local structure: [[concepts/architectures/cnn|CNN]], [[concepts/architectures/u-net|U-Net]], or [[concepts/architectures/vision-transformer|Vision Transformer]].
- Ordered text or biological sequence: [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/rnn|RNN]], or [[concepts/architectures/state-space-model|state-space model]].
- Graph-structured molecules or relational objects: [[concepts/architectures/gnn|GNN]] or [[concepts/architectures/graph-transformer|Graph Transformer]].
- Unordered sets: [[concepts/architectures/deep-sets|Deep Sets]] or [[concepts/architectures/set-transformer|Set Transformer]].
- 3D structures with symmetry constraints: [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]].
- Large conditional compute: [[concepts/architectures/mixture-of-experts|Mixture of experts]].

## Baseline Ladder

Architecture selection should not jump directly to the newest model family. A practical ladder is:

$$
\text{simple baseline}
\rightarrow
\text{modality-matched baseline}
\rightarrow
\text{inductive-bias model}
\rightarrow
\text{scaling or routed variant}
$$

| Stage | Purpose | Example |
| --- | --- | --- |
| simple baseline | catch leakage and task triviality | linear model, MLP, nearest-neighbor retrieval |
| modality baseline | respect raw input type | CNN for grid, Transformer for sequence, GNN for graph |
| bias-specific model | encode known symmetry or locality | equivariant GNN, U-Net, graph transformer |
| scaling variant | improve capacity or context under budget | long-context attention, SSM, MoE |

If the simple baseline is already strong, architecture claims need stronger leakage checks and error analysis.

## Cost Decomposition

Selection is also a systems decision:

$$
C(a)
= C_{\mathrm{train}}(a)
+ C_{\mathrm{inference}}(a)
+ C_{\mathrm{data}}(a)
+ C_{\mathrm{maintenance}}(a)
$$

| Cost | Ask |
| --- | --- |
| training | memory, batch size, optimizer state, distributed communication |
| inference | latency, throughput, batchability, cache behavior |
| data | preprocessing, graph construction, tokenization, featurization |
| maintenance | implementation complexity, dependency risk, reproducibility |

For public notes, report hardware class and measurement protocol without private hostnames or live cluster details.

## Failure Modes

| Mistake | Symptom |
| --- | --- |
| using Transformer because it is popular | weak baseline, high cost, little task-specific gain |
| using GNN without validating graph construction | performance depends on edge heuristic rather than model |
| using equivariant model when labels are not geometry-sensitive | extra complexity without measurable benefit |
| using MoE for capacity when serving latency matters | FLOPs look low but routing/all-to-all dominates |
| using CNN on coordinate grids without rotation handling | model learns frame artifacts |
| comparing architectures with different pretraining or data | architecture claim becomes a training-data claim |

## Checks

- What is the simplest architecture that should be a baseline?
- What data symmetry does the architecture encode?
- Is the selected architecture solving the task or exploiting the split?
- Does inference cost matter as much as training cost?
- Which failure mode should be tested first?
- Are objective, data, parameter count, pretraining, and compute budget matched across compared architectures?

## Related

- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/index|Evaluation]]
