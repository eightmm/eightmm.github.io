---
title: Hyperparameter Tuning
tags:
  - machine-learning
  - optimization
  - evaluation
---

# Hyperparameter Tuning

Hyperparameter tuning searches over choices that are not directly learned by gradient descent but strongly affect training and generalization. Examples include learning rate, weight decay, batch size, model width, dropout rate, augmentation strength, and scheduler settings.

A tuning problem can be written as:

$$
\lambda^\*
=
\arg\min_{\lambda \in \Lambda}
\hat{R}_{\mathrm{val}}
\left(
f_{\operatorname{Train}(\mathcal{D}_{\mathrm{train}}, \lambda)}
\right)
$$

where $\Lambda$ is the search space and $\hat{R}_{\mathrm{val}}$ is validation risk. The final claim should not use validation risk as the test estimate.

## Search Methods

- Manual tuning: small, interpretable changes guided by diagnostics.
- Grid search: exhaustive combinations over a small discrete space.
- Random search: sampled configurations, often stronger than naive grids.
- Bayesian optimization: model-based search over promising regions.
- Successive halving or bandit search: early resource allocation based on partial runs.
- Population-based training: jointly changes weights and hyperparameters during training.

Each method changes the selection budget and must be recorded when it affects the final model.

## Common Hyperparameters

| Area | Examples | Main risk |
|---|---|---|
| Optimization | learning rate, warmup, schedule, optimizer | unstable or slow training |
| Regularization | weight decay, dropout, label smoothing | overfit or underfit |
| Data | batch size, sampling, augmentation | biased gradient or distribution mismatch |
| Architecture | depth, width, heads, hidden size | capacity and compute tradeoff |
| Selection | early-stopping patience, checkpoint rule | validation overuse |
| Inference | threshold, temperature, decoding | metric mismatch or calibration drift |

## Budget and Overfitting

If many trials are evaluated on the same validation set, tuning can overfit validation:

$$
\lambda^\*
=
\arg\min_{\lambda \in \Lambda}
\left[
R(f_\lambda)
+
\epsilon_{\lambda,\mathrm{val}}
\right]
$$

The selected $\lambda^\*$ can exploit noise $\epsilon_{\lambda,\mathrm{val}}$. A larger search space should therefore be paired with a stronger final test boundary or nested validation design.

## Checks

- Is the search space documented before interpreting results?
- Are random seeds, failed trials, and early-stopped trials included in the selection story?
- Are preprocessing parameters fit only inside the train split for each trial?
- Does the tuning metric match the final deployment metric?
- Is the final test set untouched until the selected configuration is fixed?
- Is compute budget comparable when comparing methods?

## Related

- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
