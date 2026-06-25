---
title: Curriculum Learning
tags:
  - curriculum-learning
  - training
  - machine-learning
---

# Curriculum Learning

Curriculum learning changes the order or weighting of training examples so the model sees easier, cleaner, shorter, or more reliable examples before harder ones.

Instead of optimizing over a fixed data distribution, training uses a time-dependent sampling distribution:

$$
\min_\theta
\mathbb{E}_{(x,y)\sim q_t(x,y)}
\left[
\mathcal{L}(f_\theta(x),y)
\right]
$$

where $q_t$ changes with training step $t$.

An example-weighted objective is:

$$
\mathcal{L}_t(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
w_i(t)\,
\mathcal{L}(f_\theta(x_i),y_i)
$$

where $w_i(t)$ controls when and how strongly example $i$ contributes.

## Curriculum Signals

- Sequence length or context length.
- Label confidence or annotation quality.
- Data source reliability.
- Task difficulty or synthetic-to-real transition.
- Coarse-to-fine resolution.
- Simple-to-complex tool or reasoning traces.

## Why It Matters

- Can stabilize training on noisy or heterogeneous data.
- Makes data ordering a modeling decision, not just an implementation detail.
- Can bias the model if difficult or rare examples are under-sampled.
- In agent and scientific workflows, curriculum can separate basic format learning from hard verification tasks.

## Checks

- What defines easy and hard examples?
- Is the curriculum fixed, learned, or performance-adaptive?
- Are rare but important examples delayed too long?
- Does the final evaluation cover the full difficulty distribution?
- Is improvement from curriculum or simply from data filtering?

## Related

- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/evaluation/robustness|Robustness]]
