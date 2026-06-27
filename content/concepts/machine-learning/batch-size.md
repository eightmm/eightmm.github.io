---
title: Batch Size
tags:
  - machine-learning
  - optimization
  - training
---

# Batch Size

Batch size is the number of examples used to estimate the training gradient per update. It affects memory use, optimization noise, throughput, and generalization.

For a batch $B$, the mini-batch loss is:

$$
\mathcal{L}_B
=
\frac{1}{|B|}
\sum_{i\in B}
\mathcal{L}(f_\theta(x_i),y_i)
$$

With distributed training and gradient accumulation, effective batch size is:

$$
B_{\mathrm{eff}}
=
B_{\mathrm{per\ device}}
\times
N_{\mathrm{devices}}
\times
N_{\mathrm{accum}}
$$

See [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]] for the difference between micro-batches, optimizer steps, and consumed samples.

## Example Batch vs Token Batch

For fixed-size examples, example count is usually enough. For sequence, graph, or structure data, valid elements may vary:

$$
T_{\mathrm{eff}}
=
\sum_{i\in B}
\operatorname{valid\_tokens}(x_i)
$$

or more generally:

$$
N_{\mathrm{valid}}
=
\sum_{i\in B}
\sum_{j}
m_{ij}
$$

where $m_{ij}\in\{0,1\}$ is a mask for valid elements. A batch of 8 long sequences can contain more learning signal and memory pressure than a batch of 32 short sequences.

## Gradient Noise Scale

Batch size controls the stochastic gradient estimate:

$$
g_B
=
\frac{1}{B}
\sum_{i=1}^{B}
g_i
$$

With approximately independent examples:

$$
\operatorname{Var}(g_B)
\approx
\frac{1}{B}
\operatorname{Var}(g_i)
$$

This is why larger batches can make optimization smoother, while smaller batches add noise. The best batch size is a systems-and-optimization choice, not only a memory setting.

## Learning Rate Coupling

Changing effective batch size changes update dynamics. A common heuristic is linear scaling:

$$
\eta_{\mathrm{new}}
=
\eta_{\mathrm{old}}
\frac{B_{\mathrm{new}}}{B_{\mathrm{old}}}
$$

but this is not a law. Warmup, optimizer choice, normalization, gradient clipping, and task distribution can make the relationship weaker.

## Systems Boundary

Batch size is constrained by several memory terms:

$$
M_{\mathrm{train}}
\approx
M_{\mathrm{params}}
+
M_{\mathrm{grads}}
+
M_{\mathrm{optimizer}}
+
M_{\mathrm{activations}}(B)
$$

Activation memory usually grows with batch size and sequence length. If memory is the limit, the alternatives are gradient accumulation, activation checkpointing, shorter context, lower precision, smaller model, or different batching.

## Checks

- Is the reported batch size per device, global, or effective?
- Are gradients accumulated intentionally?
- Are accumulation steps included in the effective batch size calculation?
- Does the learning rate change with effective batch size?
- Does padding or variable-size data change the real number of valid elements?
- Is memory pressure from activations or optimizer state limiting batch size?
- Are schedules, validation intervals, and checkpoints counted in optimizer steps or consumed samples?
- Is throughput reported as examples/sec, tokens/sec, or optimizer steps/sec?
- Is evaluation using the same natural distribution even when training batches are balanced?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/distributed-training|Distributed training]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[infra/gpu/index#memory|GPU memory]]
