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

## Key Ideas

- Mamba belongs to the [[concepts/architectures/state-space-model|state-space model]] family but makes the state update input-dependent.
- Selective updates let the model decide what to keep, forget, or emphasize as it scans a sequence.
- The architecture is often studied as an efficient alternative or complement to [[concepts/architectures/transformer|Transformers]] for long contexts.
- It is especially relevant when sequences are long and full attention cost is a practical bottleneck.
- In protein modeling notes, treat Mamba-style modules as sequence mixers unless the paper adds explicit structure or geometry.

## Practical Checks

- Check whether the model is purely sequence-based or combined with attention, convolution, graphs, or structural features.
- Track whether outputs are token-level states, pooled representations, or generative logits.
- Look for how bidirectional context is handled when the task is not causal generation.
- For protein papers, separate architectural claims from dataset, split, and evaluation choices.

## Related

- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/transformer|Transformer]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
