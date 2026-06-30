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

## Curriculum Contract

Curriculum learning needs an explicit policy for ordering or weighting examples:

$$
w_i(t)=\pi_{\mathrm{cur}}(x_i,y_i,t;\ s_i)
$$

where $s_i$ is a difficulty, quality, source, or confidence signal.

| Contract field | Question |
| --- | --- |
| difficulty signal | what makes an example easy or hard? |
| schedule | how does sampling change with training step? |
| coverage | when does the model see the full target distribution? |
| adaptation | is the curriculum fixed or based on model performance? |
| evaluation | is the final test distribution unchanged? |

Curriculum is a training intervention, not a reason to evaluate only on easy examples.

## Schedule Types

| Type | Example | Risk |
| --- | --- | --- |
| fixed schedule | short sequences before long sequences | hand-designed difficulty may be wrong |
| source schedule | synthetic or curated data before noisy real data | source bias can persist |
| confidence schedule | high-confidence labels first | rare/ambiguous cases delayed |
| self-paced | model selects examples it can learn | confirmation bias |
| anti-curriculum | hard examples first | instability if model lacks basics |

## Evaluation Boundary

The claim should distinguish faster training from better final generalization.

| Claim | Evidence |
| --- | --- |
| stabilizes optimization | fewer divergences, smoother learning curves |
| improves sample efficiency | same metric with fewer examples/steps |
| improves final quality | better metric on full target distribution |
| improves robustness | better hard-slice or OOD performance |
| only filters bad data | compare against data cleaning baseline |

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
- Is the curriculum schedule documented enough to reproduce?
- Are easy/hard slices reported separately?

## Related

- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/evaluation/robustness|Robustness]]
