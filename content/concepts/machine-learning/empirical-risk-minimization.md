---
title: Empirical Risk Minimization
tags:
  - machine-learning
  - optimization
---

# Empirical Risk Minimization

Empirical risk minimization is the basic training principle of choosing parameters that minimize average loss on observed data.

Given a dataset:

$$
\mathcal{D}
= \{(x_i, y_i)\}_{i=1}^{n}
$$

the empirical risk is:

$$
\hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

Training chooses:

$$
\hat{\theta}
=
\arg\min_{\theta}
\hat{R}(\theta)
$$

$x_i$ is an input, $y_i$ is a target, $f_\theta$ is the model, and $\mathcal{L}$ is the loss.

## Why It Matters

Many learning methods are variations on this template. The difference is usually in the data distribution, loss, regularizer, sampling process, or optimization method.

ERM is an approximation to population risk:

$$
R(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\mathcal{L}(f_\theta(x),y)
\right]
$$

The training set replaces the unknown data distribution with an empirical distribution:

$$
\hat{p}_{\mathcal{D}}(x,y)
=
\frac{1}{n}
\sum_{i=1}^{n}
\delta_{(x_i,y_i)}(x,y)
$$

so $\hat{R}(\theta)$ is only as meaningful as the dataset, labels, preprocessing, and split policy behind it.

With a regularizer:

$$
\hat{\theta}
=
\arg\min_{\theta}
\left[
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
+ \lambda \Omega(\theta)
\right]
$$

$\Omega(\theta)$ penalizes model complexity or parameter size, and $\lambda$ controls the penalty strength.

## Weighted Risk

Real training objectives often use weights, masks, or sampling rules:

$$
\hat{R}_{w}(\theta)
=
\frac{
\sum_{i=1}^{n}
w_i
\mathcal{L}(f_\theta(x_i), y_i)
}{
\sum_{i=1}^{n}
w_i
}
$$

where $w_i$ may encode class weights, label confidence, missing-label masks, token masks, or task balancing. This changes the target empirical distribution. A model trained on reweighted risk should not be evaluated as if it optimized the unweighted population.

## Protocol Boundary

ERM is not only a formula. It includes decisions that must be fixed before final evaluation:

$$
\text{training protocol}
=
(\mathcal{D}_{\mathrm{train}},
\text{preprocessing},
\mathcal{L},
\Omega,
\text{sampler},
\text{optimizer},
\text{selection rule})
$$

Changing any component after seeing validation or test behavior changes the learning procedure. This connects ERM to [[concepts/machine-learning/model-selection|model selection]] and [[concepts/evaluation/leakage|leakage]].

## Checks

- What examples define the empirical distribution?
- Does the loss match the task and target semantics?
- Is the training objective the same quantity as the evaluation metric?
- Are weights, masks, or sampling probabilities changing the risk?
- Is regularization explicit or hidden in the optimizer?
- Which parts of preprocessing are fit only on training data?
- Is the final test set untouched by objective, sampler, or hyperparameter choices?
- Does the claimed deployment distribution match the empirical distribution used for training?

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/generalization|Generalization]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
