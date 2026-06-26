---
title: Computational complexity
tags:
  - architectures
  - systems
---

# Computational Complexity

Computational complexity describes how memory, compute, or communication cost changes as input size and model width grow.

For an architecture $a$:

$$
C_a(n, d) = O(g_a(n, d))
$$

$n$ is a size variable such as sequence length, number of pixels, or number of graph edges, $d$ is hidden dimension, and $g_a$ is the dominant scaling term.

## Common Patterns

Dense linear layer from $d_{\mathrm{in}}$ to $d_{\mathrm{out}}$:

$$
y = xW + b,
\qquad
C = O(d_{\mathrm{in}} d_{\mathrm{out}})
$$

Self-attention with sequence length $L$ and hidden dimension $d$:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),
\qquad
Y = AV
$$

The attention matrix $QK^\top$ costs $O(L^2 d_k)$ and stores $O(L^2)$ attention weights before implementation details.

2D convolution over height $H$, width $W$, kernel size $k$, input channels $C_{\mathrm{in}}$, and output channels $C_{\mathrm{out}}$:

$$
C = O(HWk^2 C_{\mathrm{in}} C_{\mathrm{out}})
$$

Graph message passing with edge set $E$:

$$
C = O(|E|d)
$$

for a simple per-edge message of hidden size $d$.

## Why It Matters

Architecture choice is also a systems decision. A model can be statistically reasonable but impossible to train, too slow to serve, or too memory-heavy for long inputs.

Complexity is not the whole story. Constants, kernels, hardware layout, batching, parallelism, and memory bandwidth can dominate in practice.

For paper claims, complexity should be separated from the actual scaling evidence. A lower asymptotic term does not prove better quality per compute unless the budget, implementation, and metric are comparable.

## Checks

- What variable grows at deployment: sequence length, graph size, image resolution, batch size, or number of candidates?
- Is the bottleneck arithmetic, activation memory, KV cache, communication, or IO?
- Does the benchmark hide the real deployment size?
- Is the asymptotic cost compatible with the intended inference workflow?
- Is a scaling claim backed by matched data, model size, compute, hardware, and metric?

## Related

- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
