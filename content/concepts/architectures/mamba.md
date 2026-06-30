---
title: Mamba
tags:
  - architectures
  - mamba
  - state-space-model
---

# Mamba

Mamba is a selective state-space architecture for sequence modeling. In this wiki, it is mainly a seed topic for efficient long-context models and protein modeling experiments.

A generic selective state update can be viewed as:

$$
h_t = A(x_t)h_{t-1} + B(x_t)x_t,
\qquad
y_t = C(x_t)h_t
$$

The key distinction is that update parameters depend on the current input $x_t$.

A more implementation-facing abstraction is:

$$
\Delta_t, B_t, C_t = g(x_t)
$$

$$
h_t = \bar{A}(\Delta_t)h_{t-1} + \bar{B}(\Delta_t, B_t)x_t,
\qquad
y_t = C_t h_t
$$

where $g$ is a learned projection and $\Delta_t$ controls the input-dependent discretization or update scale.

## State-Space Boundary

Mamba should be read as a member of the [[concepts/architectures/state-space-model|state-space model]] family:

$$
h_t = F_t h_{t-1} + G_t x_t,
\qquad
y_t = H_t h_t
$$

The selective part means $F_t,G_t,H_t$ can depend on the input token. This differs from a fixed linear time-invariant SSM, where transition matrices are shared across positions.

## Scan View

The recurrence can be evaluated as a scan over sequence positions:

$$
h_t
=
\left(\prod_{i=1}^{t}F_i\right)h_0
+
\sum_{j=1}^{t}
\left(\prod_{i=j+1}^{t}F_i\right)G_jx_j
$$

Implementations exploit associative scan-style structure rather than treating every step as an independent matrix multiplication. The practical claim is efficient long-sequence mixing with linear or near-linear scaling in sequence length.

## Structured State Space Duality

[[papers/architectures/mamba-2|Mamba-2]] adds a matrix-mixer view of the family. A selective SSM recurrence can be expanded into a causal mixing matrix:

$$
y_t
=
\sum_{i\le t}
C_t
\left(
\prod_{j=i+1}^{t}A_j
\right)
B_i x_i.
$$

This matters because the same sequence transformation can sometimes be viewed as recurrence, structured matrix multiplication, or attention-like mixing. For paper reading, this is the bridge between [[concepts/architectures/state-space-model|state-space models]] and [[concepts/architectures/attention|attention]].

## Causal and Bidirectional Use

A left-to-right Mamba block is naturally causal:

$$
y_t = f(x_{\le t})
$$

For classification, protein representation, or structure-related tasks, papers may add bidirectional scanning, pooling, attention, convolution, or task-specific heads:

$$
z = \operatorname{pool}(Y_{\rightarrow}, Y_{\leftarrow})
$$

This design choice matters because a causal sequence model and a bidirectional representation model have different information access.

## Key Ideas

- Mamba belongs to the [[concepts/architectures/state-space-model|state-space model]] family but makes the state update input-dependent.
- Selective updates let the model decide what to keep, forget, or emphasize as it scans a sequence.
- The architecture is often studied as an efficient alternative or complement to [[concepts/architectures/transformer|Transformers]] for long contexts.
- Hybrid backbones such as [[papers/architectures/jamba|Jamba]] use Mamba layers alongside attention and MoE rather than treating Mamba as an all-or-nothing Transformer replacement.
- It is especially relevant when sequences are long and full attention cost is a practical bottleneck.
- In protein modeling notes, treat Mamba-style modules as sequence mixers unless the paper adds explicit structure or geometry.

## Practical Checks

- Check whether the model is purely sequence-based or combined with attention, convolution, graphs, or structural features.
- Track whether outputs are token-level states, pooled representations, or generative logits.
- Look for how bidirectional context is handled when the task is not causal generation.
- For protein papers, separate architectural claims from dataset, split, and evaluation choices.
- Check whether the paper compares against attention under matched parameter count, context length, and training tokens.
- Check whether long-context benefit is due to architecture, data curriculum, implementation, or evaluation setting.

## Related

- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[papers/architectures/jamba|Jamba]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
