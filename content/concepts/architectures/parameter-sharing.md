---
title: Parameter sharing
tags:
  - architectures
  - machine-learning
---

# Parameter Sharing

Parameter sharing means reusing the same learnable parameters across positions, tokens, nodes, edges, patches, or time steps.

For a generic local update:

$$
h_i' = \phi_\theta(h_i, c_i)
$$

$h_i$ is the representation at location $i$, $c_i$ is context around that location, and the same function $\phi_\theta$ is reused for many $i$.

## Convolution Example

For a 1D convolution:

$$
y_i = \sum_{k=-r}^{r} w_k x_{i+k} + b
$$

The same kernel weights $w_k$ are applied at every position $i$. This creates translation equivariance and reduces the number of parameters compared with a fully connected map.

## Graph Message Passing Example

For message passing:

$$
m_i = \sum_{j \in \mathcal{N}(i)} \phi_\theta(h_i, h_j, e_{ij})
$$

$$
h_i' = \psi_\theta(h_i, m_i)
$$

$\mathcal{N}(i)$ is the neighborhood of node $i$, $e_{ij}$ is an edge feature, and the same message/update functions are reused across graph locations.

## Why It Matters

- It reduces parameter count.
- It improves generalization when repeated structure exists.
- It encodes [[concepts/architectures/inductive-bias|inductive bias]] such as locality, recurrence, or permutation equivariance.
- It makes models more reusable across variable input sizes.

## Examples

- [[concepts/architectures/cnn|CNN]] kernels share weights across image or grid positions.
- [[concepts/architectures/rnn|RNN]] transition functions share weights across time.
- [[concepts/architectures/transformer|Transformer]] attention projections share weights across token positions.
- [[concepts/architectures/gnn|GNN]] message functions share weights across nodes and edges.
- [[concepts/architectures/mixture-of-experts|Mixture of experts]] shares routers and experts conditionally, but activates only a subset per token.

## Checks

- What axis uses shared parameters?
- Does sharing match the true repeated structure of the input?
- Does the model need positional or edge information to break unwanted symmetry?
- Does sharing reduce capacity too much for the task?

## Related

- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/architectures/linear-layer|Linear layer]]
