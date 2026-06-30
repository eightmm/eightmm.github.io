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

## Shift Types

Different shifts require different adaptation evidence.

| Shift Type | Informal Meaning | Common Adaptation Need |
| --- | --- | --- |
| covariate shift | input distribution changes, label rule roughly stable | reweighting, target-domain pretraining, robust features |
| label shift | class or target prevalence changes | calibration, prior correction, threshold changes |
| concept shift | label rule changes | target labels or task redefinition |
| support shift | target contains regions unseen in source | stronger target data and uncertainty checks |
| protocol shift | measurement or annotation process changes | harmonization and source-aware evaluation |

In notation:

$$
p_s(x)\ne p_t(x)
$$

is different from:

$$
p_s(y\mid x)\ne p_t(y\mid x)
$$

The second case is harder because unlabeled target examples do not directly reveal the new label rule.

## Adaptation Data Regimes

| Target Data Available | Typical Method | Main Risk |
| --- | --- | --- |
| none | robustness or source-only generalization | target claim is weak |
| unlabeled target inputs | continued pretraining, alignment, normalization update | target leakage into model selection |
| small labeled target set | fine-tuning, adapters, calibration | overfitting and search-budget inflation |
| streaming target feedback | active learning or continual adaptation | non-stationary evaluation |
| source and target labels | supervised domain adaptation | source/target split must be explicit |

The validation set should reflect the target claim. If target validation examples guide adaptation, a separate final target test set is needed.

## Source-Target Risk

Source performance and target performance should be tracked separately:

$$
R_s(f)
=
\mathbb{E}_{p_s}
[\mathcal{L}(f(x),y)],
\qquad
R_t(f)
=
\mathbb{E}_{p_t}
[\mathcal{L}(f(x),y)]
$$

An adaptation method can improve $R_t$ while degrading $R_s$. That may be acceptable, but it should be reported as specialization rather than universal improvement.

## Checks

- What exactly differs between source and target: input, label, protocol, species, time, instrument, or task?
- Are target validation examples independent from adaptation data?
- Does adaptation improve target performance without hiding source-domain regressions?
- Is the target domain broad enough for the intended deployment claim?
- Is the shift covariate, label, concept, support, protocol, or a mixture?
- Are target test examples untouched by adaptation and hyperparameter selection?
- Is source performance still relevant, and if so is it reported?

## Related

- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
