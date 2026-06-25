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

## Checks

- Is the reported batch size per device, global, or effective?
- Are gradients accumulated intentionally?
- Are accumulation steps included in the effective batch size calculation?
- Does the learning rate change with effective batch size?
- Does padding or variable-size data change the real number of valid elements?
- Is memory pressure from activations or optimizer state limiting batch size?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[infra/distributed-training|Distributed training]]
- [[infra/gpu-memory|GPU memory]]
