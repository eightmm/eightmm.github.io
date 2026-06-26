---
title: Data Distribution
tags:
  - data
  - probability
  - machine-learning
---

# Data Distribution

A data distribution describes how examples and labels are generated. A dataset is a finite sample, but machine learning claims are usually about the distribution the sample represents.

For supervised learning, the population risk is:

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\mathcal{L}(f(x),y)
\right]
$$

where $p_{\mathrm{data}}(x,y)$ is the data-generating distribution.

The empirical dataset approximates this distribution:

$$
\hat{p}_{\mathcal{D}}(x,y)
=
\frac{1}{n}
\sum_{i=1}^{n}
\delta(x=x_i,y=y_i)
$$

## Training and Deployment

The core generalization question is whether training and deployment distributions match:

$$
p_{\mathrm{train}}(x,y)
\approx
p_{\mathrm{deploy}}(x,y)
$$

If they differ, validation performance may not estimate deployment behavior.

## What Defines a Distribution

- Input source and collection process.
- Inclusion and exclusion criteria.
- Label generation protocol.
- Measurement noise and censoring.
- Time, location, instrument, batch, species, target, or task context.
- Missing data and unobserved negative examples.

## Factorization

A supervised distribution can be factorized as:

$$
p(x,y)
=
p(y\mid x)p(x)
$$

or, when metadata matters:

$$
p(x,y,m)
=
p(y\mid x,m)p(x\mid m)p(m)
$$

where $m$ may include source, time, assay, domain, user, device, scaffold, family, or other grouping context. Distribution shift can affect any part of this factorization.

## Split Distributions

Train, validation, test, and deployment distributions should be named:

$$
p_{\mathrm{train}},
\quad
p_{\mathrm{val}},
\quad
p_{\mathrm{test}},
\quad
p_{\mathrm{deploy}}
$$

Validation estimates are useful only if $p_{\mathrm{val}}$ supports the model-selection decision, while test estimates matter only if $p_{\mathrm{test}}$ matches the claimed generalization target.

## Failure Modes

- The dataset is treated as IID even though examples share groups or sources.
- The test set has a different label prevalence than deployment.
- Source-specific artifacts dominate the input distribution.
- Missing labels are mistaken for true negatives.
- The model is evaluated on a curated distribution but deployed on raw data.

## Checks

- What population should the dataset represent?
- What examples are systematically absent?
- Are labels sampled from the same protocol across splits?
- Does the test distribution match the intended claim?
- Are multiple data sources pooled without source-aware evaluation?
- Are train, validation, test, and deployment distributions described separately?
- Is the shift in $p(x)$, $p(y\mid x)$, metadata, or label availability?
- Are subgroup distributions reported for important domains or sources?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
