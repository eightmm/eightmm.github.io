---
title: Seed Variance
tags:
  - evaluation
  - reproducibility
  - statistics
---

# Seed Variance

Seed variance is the variation in a result caused by random choices in training, data order, initialization, sampling, augmentation, decoding, or evaluation. It is a reproducibility boundary for model comparisons.

For scores from $K$ independent runs:

$$
\bar{M}
=
\frac{1}{K}\sum_{k=1}^{K} M_k,
\qquad
s_M^2
=
\frac{1}{K-1}\sum_{k=1}^{K}(M_k-\bar{M})^2
$$

where $M_k$ is the metric from seed or run $k$.

## Why It Matters

A reported improvement is weak when it is smaller than run-to-run variation:

$$
\Delta
=
\bar{M}_A-\bar{M}_B
$$

If $|\Delta|$ is comparable to $s_M$, the paper should not be read as strong evidence that one method is better without a paired or repeated-run analysis.

## Difference Distribution

For paired runs where both methods share seed or split $k$:

$$
d_k=M_{A,k}-M_{B,k}
$$

summarize:

$$
\bar{d}
=
\frac{1}{K}\sum_{k=1}^{K}d_k,
\qquad
s_d^2
=
\frac{1}{K-1}\sum_{k=1}^{K}(d_k-\bar{d})^2
$$

This is usually more informative than comparing two independent standard deviations because it controls shared randomness.

## Confidence Interval

A simple repeated-run interval for the mean score is:

$$
\bar{M}
\pm
t_{K-1,1-\alpha/2}
\frac{s_M}{\sqrt{K}}
$$

For method differences, use the paired differences $d_k$:

$$
\bar{d}
\pm
t_{K-1,1-\alpha/2}
\frac{s_d}{\sqrt{K}}
$$

With very small $K$, this interval is noisy; report the individual run values as well.

## Sources

| Source | Example |
| --- | --- |
| Initialization | random weights or adapter initialization |
| Data order | mini-batch ordering and shuffling |
| Sampling | negative sampling, augmentation, diffusion sampling, decoding |
| Split construction | random split or fold assignment |
| Hardware/numerics | nondeterministic kernels, mixed precision, reduction order |
| Model selection | best checkpoint, best prompt, best seed, best threshold |

## Reporting Contract

| Field | Requirement |
| --- | --- |
| Seed policy | number of seeds, fixed seed list, or reason only one seed is used |
| Run unit | what changes between runs: initialization, data order, split, prompt, sampler |
| Summary | mean and standard deviation, confidence interval, or paired difference |
| Selection rule | whether the best, last, or validation-selected checkpoint is reported |
| Failed runs | whether crashed, diverged, or excluded runs are counted |
| Compute budget | whether all compared methods received the same number of attempts |

## Best-Seed Trap

Reporting the best seed estimates a different quantity:

$$
\max_k M_k
\neq
\mathbb{E}[M]
$$

If one method receives more attempts, it has a higher chance of a lucky run. Compare equal attempt budgets or report both mean and best-seed behavior explicitly.

## Checks

- Is the reported value mean, median, best seed, or a single run?
- Are all compared methods run with the same seed policy?
- Is the improvement larger than run-to-run variation?
- Are failed or unstable runs included in the summary?
- Is variance over seeds, data splits, prompts, decoders, or test examples?
- Was the test set used repeatedly to choose a seed or checkpoint?
- Are method differences paired by seed or split where possible?
- Is the number of attempts equal across compared methods?

## Related

- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[papers/analysis/result-table-reading|Result table reading]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
