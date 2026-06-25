---
title: Time-Series Forecasting
tags:
  - tasks
  - forecasting
  - time-series
---

# Time-Series Forecasting

Time-series forecasting predicts future values from past observations. The input is ordered by time, and evaluation must respect temporal causality.

For a sequence $x_{1:T}$, a forecasting model predicts a future horizon:

$$
\hat{x}_{T+1:T+H}
=
f_\theta(x_{1:T}, c)
$$

where $H$ is the forecast horizon and $c$ represents optional covariates such as calendar features, conditions, interventions, or external signals.

The supervised loss for a horizon can be:

$$
\mathcal{L}
=
\frac{1}{H}
\sum_{h=1}^{H}
\ell(\hat{x}_{T+h}, x_{T+h})
$$

## Key Ideas

- Forecasting differs from generic regression because training and evaluation must preserve time order.
- Short-horizon and long-horizon forecasts can require different models and metrics.
- Covariates may be known in advance, observed only in the past, or unavailable at deployment.
- Autoregressive rollouts can accumulate error when predictions feed future inputs.
- Temporal leakage is common when preprocessing or split construction uses future information.

## Practical Checks

- What is the forecast horizon?
- Are splits chronological rather than randomly shuffled?
- Are all covariates available at prediction time?
- Is performance reported per horizon or only averaged?
- Does the baseline include persistence, seasonal, or simple statistical forecasts?

## Related

- [[concepts/modalities/tabular|Tabular]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
