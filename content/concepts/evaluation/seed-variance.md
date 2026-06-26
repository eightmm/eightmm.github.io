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

## Checks

- Is the reported value mean, median, best seed, or a single run?
- Are all compared methods run with the same seed policy?
- Is the improvement larger than run-to-run variation?
- Are failed or unstable runs included in the summary?
- Is variance over seeds, data splits, prompts, decoders, or test examples?
- Was the test set used repeatedly to choose a seed or checkpoint?

## Related

- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[papers/analysis/result-table-reading|Result table reading]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
