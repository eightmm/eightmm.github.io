---
title: Sampling Bias
tags:
  - data
  - sampling
  - evaluation
---

# Sampling Bias

Sampling bias occurs when the observed dataset is not representative of the population or deployment distribution that the model will face. It is different from random sampling noise: the collection process systematically favors some examples and misses others.

Let $p_{\mathrm{target}}(x,y)$ be the target distribution and $q_{\mathrm{sample}}(x,y)$ be the sampled dataset distribution:

$$
q_{\mathrm{sample}}(x,y)
\ne
p_{\mathrm{target}}(x,y)
$$

The model trains and validates on $q_{\mathrm{sample}}$, but the claim is often about $p_{\mathrm{target}}$.

## Common Sources

- Convenience sampling from easy-to-collect examples.
- Missing negative labels or unobserved positives.
- Source, assay, time, site, user, scaffold, species, or protein-family coverage gaps.
- Filtering rules that remove hard or ambiguous cases.
- Duplicate-heavy datasets that overrepresent common entities.
- Active-learning or screening pipelines that bias which examples are measured.

## Selection Mechanism

Sampling bias can be written with a selection variable:

$$
S\in\{0,1\}
$$

where $S=1$ means the example enters the dataset. The observed distribution is:

$$
q(x,y)
=
p(x,y\mid S=1)
$$

If selection depends on $x$, $y$, source, time, or prior model score, the observed dataset may not support deployment claims without adjustment.

| Selection depends on | Example consequence |
| --- | --- |
| source | model learns dataset origin |
| label | positives or failures are overrepresented |
| model score | active-learning feedback loop |
| time | historical data misses current distribution |
| availability | easy cases dominate benchmark |

## Importance Weighting

If the target and sampling distributions are known or estimable, risk under the target distribution can be written as:

$$
R_{p}(f)
=
\mathbb{E}_{(x,y)\sim q}
\left[
\frac{p(x,y)}{q(x,y)}
\mathcal{L}(f(x),y)
\right]
$$

In practice, the density ratio $p/q$ is rarely known exactly, so bias must also be audited with metadata, split design, and slice metrics.

## Symptoms

- Strong validation performance but weak deployment behavior.
- Performance concentrated in overrepresented groups.
- Rare classes, rare domains, or hard negatives have poor recall.
- Calibration changes when prevalence or source changes.
- A simple metadata or source classifier predicts split membership.

## Checks

- What population should the dataset represent?
- Which examples are systematically missing?
- Are train, validation, test, and deployment distributions described separately?
- Can metadata explain performance differences across slices?
- Are duplicates and repeated entities overrepresented?
- Does the split test the bias or accidentally preserve it across all splits?
- Is the observed dataset distribution separated from the intended deployment distribution?
- Are source, time, assay, scaffold, family, or site slices reported?

## Related

- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
