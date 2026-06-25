---
title: Domain Adaptation
tags:
  - domain-adaptation
  - transfer-learning
  - generalization
---

# Domain Adaptation

Domain adaptation adjusts a model trained on a source domain to work better on a target domain. It is useful when target labels are limited but the target distribution differs from the source.

Let source and target distributions be:

$$
(x,y)\sim p_s(x,y),
\qquad
(x,y)\sim p_t(x,y)
$$

The goal is to reduce target risk:

$$
R_t(f)
=
\mathbb{E}_{(x,y)\sim p_t}
\left[
\mathcal{L}(f(x),y)
\right]
$$

using source data, limited target data, unlabeled target data, or a pretrained model.

## Common Patterns

- Fine-tune on labeled target examples.
- Continue pretraining on unlabeled target-domain data.
- Align source and target representations.
- Reweight source samples to better match the target distribution.
- Use adapters or low-rank updates for domain-specific behavior.

## Shift View

Domain adaptation is usually motivated by:

$$
p_s(x,y) \ne p_t(x,y)
$$

If $p_t(y\mid x)$ changes substantially, adaptation needs target labels or a reliable proxy. If mainly $p(x)$ changes, unlabeled target data can help.

## Checks

- What exactly differs between source and target: input, label, protocol, species, time, instrument, or task?
- Are target validation examples independent from adaptation data?
- Does adaptation improve target performance without hiding source-domain regressions?
- Is the target domain broad enough for the intended deployment claim?

## Related

- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/robustness|Robustness]]
