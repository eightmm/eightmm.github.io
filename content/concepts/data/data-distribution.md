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

## Checks

- What population should the dataset represent?
- What examples are systematically absent?
- Are labels sampled from the same protocol across splits?
- Does the test distribution match the intended claim?
- Are multiple data sources pooled without source-aware evaluation?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
