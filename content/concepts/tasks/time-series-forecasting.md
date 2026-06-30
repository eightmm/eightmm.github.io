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

## Forecasting Contract

Forecasting is defined by when the prediction is made and what information is available at that time.

$$
\hat{y}_{t+1:t+H}
=
f_\theta
\left(
x_{\le t},
c_{\le t},
c_{t+1:t+H}^{\mathrm{known}}
\right)
$$

| Field | Meaning |
| --- | --- |
| lookback window | how much past context the model sees |
| forecast origin $t$ | time at which prediction is issued |
| horizon $H$ | how far ahead the model predicts |
| target frequency | hourly, daily, event-based, irregular |
| known future covariates | calendar, planned intervention, schedule |
| unknown future covariates | measurements unavailable at prediction time |

If a covariate is only observed after the forecast origin, using it as input is temporal leakage.

## Evaluation by Horizon

An average score can hide horizon-specific behavior.

$$
\operatorname{MAE}(h)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|\hat{x}_{t_i+h}-x_{t_i+h}\right|
$$

| Report | Why |
| --- | --- |
| per-horizon metric | short and long horizon errors differ |
| aggregate metric | gives one summary number |
| baseline comparison | persistence/seasonal baseline can be strong |
| rolling-origin evaluation | tests repeated deployment-like forecasts |

## Leakage Patterns

| Leakage | Example |
| --- | --- |
| random split | future points from same series appear in train |
| global normalization | scaler fit on full time range |
| future covariate | using measurement not known at forecast time |
| target-derived feature | rolling statistic computed with future values |
| backfilled labels | correction from future data enters training input |

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
- Are normalization and feature computation fit only on past/train data?
- Does the evaluation simulate the actual forecast origin?

## Related

- [[concepts/modalities/tabular|Tabular]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
