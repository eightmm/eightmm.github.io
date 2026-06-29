---
title: Objective Taxonomy
tags:
  - learning
  - objective
---

# Objective Taxonomy

An objective taxonomy classifies the training signal before naming the model. The same architecture can learn different behavior under supervised labels, likelihood, contrastive pairs, denoising targets, flow fields, preferences, or rewards.

$$
\theta^\star = \arg\min_\theta \mathbb{E}_{u\sim p_{\mathrm{train}}}
\left[\mathcal{L}_{\mathrm{objective}}(u;\theta)\right]
$$

## Objective Families

| Family | Signal | Typical form | Main question |
| --- | --- | --- | --- |
| supervised | measured label $y$ | $\mathcal{L}(f_\theta(x), y)$ | Is the label semantic and split unit valid? |
| likelihood | observed sample $x$ | $-\log p_\theta(x)$ | What distribution is modeled? |
| masked modeling | hidden part of input | $-\log p_\theta(x_{\mathrm{mask}}\mid x_{\mathrm{visible}})$ | Does the mask preserve useful structure? |
| contrastive | positive/negative pairs | InfoNCE or ranking loss | Are positives and negatives semantically correct? |
| reconstruction | input reconstruction | $\lVert x-\hat{x}\rVert$ or likelihood | Is reconstruction aligned with downstream use? |
| denoising | clean target from noisy input | $\lVert \epsilon-\epsilon_\theta(x_t,t)\rVert^2$ | What corruption process is assumed? |
| flow matching | target velocity field | $\lVert v_\theta(x_t,t)-u_t(x_t)\rVert^2$ | What path and velocity define generation? |
| preference | chosen/rejected outputs | pairwise logistic objective | Does preference reflect target utility? |
| reinforcement learning | reward over trajectories | expected return $J(\theta)$ | Is reward aligned and stable? |

## Reading Checks

- Is the paper contribution the objective, the architecture, the data, or the evaluation protocol?
- Is the training signal available at inference time, or only during training?
- Does the objective optimize the same quantity as the reported metric?
- Are negative examples, masks, noisy samples, or preferences constructed without test leakage?
- Does the objective encourage representation collapse, shortcut learning, or proxy optimization?

## Related

- [[ai/learning-methods|Learning Methods]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/architectures/architecture-objective-fit|Architecture-objective fit]]
