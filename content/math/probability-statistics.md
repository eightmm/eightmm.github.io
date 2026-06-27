---
title: Probability and Statistics
tags:
  - math
  - probability
  - statistics
---

# Probability and Statistics

Probability describes uncertainty and data-generating processes. Statistics describes how finite samples estimate unknown quantities.

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
[\mathcal{L}(f(x), y)]
$$

The distribution under the expectation matters as much as the loss itself.

## Route Map

| Route | Use For | Start |
| --- | --- | --- |
| Random objects | what variable, event, sample space, and distribution are being modeled | [Random variable](/concepts/math/random-variable), [Probability distribution](/concepts/math/probability-distribution) |
| Expectations | loss, risk, metric, Monte Carlo estimate, sampling average | [Expectation](/concepts/math/expectation), [Monte Carlo estimation](/concepts/math/monte-carlo-estimation) |
| Common distributions | noise model, uncertainty assumption, approximate sampling law | [Normal distribution](/concepts/math/normal-distribution) |
| Conditioning and inference | posterior, likelihood, prior, Bayesian update | [Bayes rule](/concepts/math/bayes-rule), [Bayesian inference](/concepts/math/bayesian-inference) |
| Likelihood and information | NLL, cross-entropy, KL, generative objectives | [Maximum likelihood](/concepts/math/maximum-likelihood), [Entropy and KL divergence](/concepts/math/entropy-kl), [Information and likelihood](/math/information-likelihood) |
| Statistical evidence | finite-sample estimates, uncertainty, hypothesis tests, bias/variance | [Statistical estimator](/concepts/math/statistical-estimator), [Hypothesis testing](/concepts/math/hypothesis-testing), [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff) |

## Distribution Map

| Quantity | Meaning | Common AI Use |
| --- | --- | --- |
| $p(x)$ | marginal distribution of inputs | data modeling, density estimation |
| $p(y\mid x)$ | conditional label distribution | classification, regression uncertainty |
| $p_\theta(y\mid x)$ | model-predicted conditional distribution | probabilistic prediction and calibration |
| $p_{\mathrm{train}}(x,y)$ | training distribution | empirical risk minimization |
| $p_{\mathrm{test}}(x,y)$ | evaluation distribution | generalization claim |
| $q(z\mid x)$ | approximate posterior or encoder distribution | VAE, latent-variable inference |
| $p(z)$ | prior over latent variables | generative modeling and regularization |
| $p_\theta(x\mid z)$ | likelihood or decoder distribution | reconstruction and conditional generation |

Bayes rule connects posterior, likelihood, and prior:

$$
p(y\mid x)
=
\frac{p(x\mid y)p(y)}{p(x)}
$$

This is useful when reading probabilistic classifiers, latent-variable models, and uncertainty notes.

## Conditional Modeling

Most supervised AI claims can be written as a conditional distribution:

$$
p_\theta(y\mid x)
$$

Training then chooses parameters that make observed labels likely under the model:

$$
\theta^\star
=
\operatorname*{arg\,max}_\theta
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

For classification with logits $z=f_\theta(x)$:

$$
p_\theta(y=k\mid x)
=
\frac{\exp z_k}{\sum_{j=1}^{K}\exp z_j}
$$

and the negative log-likelihood for one example is:

$$
\mathcal{L}_{\mathrm{CE}}
=
-\log p_\theta(y\mid x)
$$

This is why cross-entropy, maximum likelihood, calibration, and uncertainty are one connected topic rather than separate tricks.

## Distribution Shift

Generalization claims depend on which distribution the expectation refers to:

$$
R_{\mathrm{deploy}}(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{deploy}}}
\left[\mathcal{L}(f_\theta(x),y)\right]
$$

The empirical test estimate is only evidence for deployment if:

$$
p_{\mathrm{test}}(x,y)
\approx
p_{\mathrm{deploy}}(x,y)
$$

In computational biology this can fail through scaffold shift, protein-family shift, assay/source shift, structure-source shift, or negative-set construction.

## Statistics

Statistics turns finite observations into claims about a population quantity. Correlation, uncertainty, hypothesis tests, and benchmark estimates should be read as estimates with assumptions, not as exact facts.

| Topic | Start |
| --- | --- |
| Covariation | [Covariance and correlation](/concepts/math/covariance-correlation) |
| Asymptotic approximation | [Central limit theorem](/concepts/math/central-limit-theorem) |
| Evidence against a null | [Hypothesis testing](/concepts/math/hypothesis-testing) |
| Error decomposition | [Bias-variance tradeoff](/concepts/math/bias-variance-tradeoff) |
| Sampling estimate | [Monte Carlo estimation](/concepts/math/monte-carlo-estimation) |

## Estimation

A finite dataset gives an estimate of a population quantity:

$$
\mu = \mathbb{E}_{X\sim p}[h(X)],
\qquad
\hat{\mu}
=
\frac{1}{n}\sum_{i=1}^{n} h(x_i)
$$

The error is controlled by sampling noise, bias, dependence between samples, and whether the sample distribution matches the target distribution.

For an estimator $\hat{\theta}$:

$$
\operatorname{MSE}(\hat{\theta})
=
\operatorname{Var}(\hat{\theta})
+
\operatorname{Bias}(\hat{\theta})^2
$$

## AI Connections

- Probabilistic prediction needs calibrated probabilities, not only scores.
- Dataset shift changes the distribution under the expected risk.
- Uncertainty estimation depends on what randomness is being modeled.
- Bayesian inference separates prior, likelihood, posterior, and posterior predictive claims.
- Hypothesis testing and confidence intervals help interpret benchmark differences.

## Computational Biology Connections

| Question | Statistical Object | Common Risk |
| --- | --- | --- |
| Is an activity label reliable? | noisy observation of an assay endpoint | assay protocol and unit differences |
| Does a split test transfer? | estimate under a target distribution | scaffold, family, or source leakage |
| Is a screening score useful? | ranking statistic over a candidate pool | active prevalence and decoy bias |
| Is uncertainty meaningful? | predictive distribution or interval | calibrated on the wrong domain |
| Is a generative model valid? | sample distribution $p_\theta(x)$ | validity without utility or novelty |

## Checks

- Is the probability conditional or marginal?
- Is the estimate biased, high-variance, or data-leaking?
- Is the test distribution the same as the deployment distribution?
- Are repeated evaluations creating multiple-comparison risk?
- Is the claim about likelihood, decision quality, ranking, calibration, or downstream utility?
- Does the reported metric estimate the same expectation the text claims?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[ai/machine-learning|Machine learning]]
- [[molecular-modeling/data-evaluation|Computational Biology data and evaluation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
