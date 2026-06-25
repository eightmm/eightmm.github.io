---
title: Model Selection
tags:
  - machine-learning
  - evaluation
  - generalization
---

# Model Selection

Model selection is the process of choosing one trained model, checkpoint, architecture, preprocessing pipeline, or hyperparameter setting before the final test claim. It is part of the learning procedure, not an afterthought.

Given candidate configurations $\lambda \in \Lambda$, training produces a fitted model:

$$
\hat{\theta}_{\lambda}
=
\operatorname{Train}(\mathcal{D}_{\mathrm{train}}, \lambda)
$$

Selection should use validation evidence:

$$
\lambda^\*
=
\arg\min_{\lambda \in \Lambda}
\hat{R}_{\mathrm{val}}(f_{\hat{\theta}_{\lambda}})
$$

The final test result then evaluates the fixed choice:

$$
\hat{R}_{\mathrm{test}}
\left(
f_{\hat{\theta}_{\lambda^\*}}
\right)
$$

Here $\lambda$ can include architecture size, learning rate, regularization strength, preprocessing choices, checkpoint epoch, augmentation policy, threshold, or decoding setting.

## What Counts as Selection

Selection includes any choice influenced by validation or test-like feedback:

- Choosing the architecture family or model size.
- Choosing hyperparameters and optimizer settings.
- Choosing a checkpoint or early-stopping epoch.
- Choosing preprocessing, augmentation, or feature sets.
- Choosing thresholds, calibration parameters, or decoding settings.
- Choosing which failed runs to exclude.
- Choosing which metric becomes primary after seeing results.

If the choice used performance feedback, it belongs in the protocol.

## Selection Bias

Trying many candidates increases the chance of selecting a model that looks good by noise. A simple view is:

$$
\hat{R}_{\mathrm{val}}(f_\lambda)
=
R(f_\lambda)
+
\epsilon_\lambda
$$

where $\epsilon_\lambda$ is validation noise. As $|\Lambda|$ grows, the selected candidate can have unusually favorable $\epsilon_\lambda$ even if its true risk is not best.

## Protocol

For a public result, record:

- Candidate set $\Lambda$ or search space.
- Training rule for each candidate.
- Validation metric used for selection.
- Selection budget: number of trials, seeds, folds, or sweeps.
- Final fixed model identity.
- Whether the test set was used exactly once after selection.

## Checks

- Was the primary selection metric defined before tuning?
- Were all candidates trained under the same data split and preprocessing contract?
- Was the test set untouched during architecture, checkpoint, and hyperparameter choices?
- Are failed or discarded runs accounted for in the narrative?
- Does the validation split match the intended [[concepts/machine-learning/generalization|generalization]] claim?
- Is the final model reproducible from the recorded configuration and checkpoint?

## Related

- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/generalization|Generalization]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
