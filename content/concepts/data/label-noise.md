---
title: Label Noise
tags:
  - data
  - labels
  - evaluation
---

# Label Noise

Label noise occurs when target labels are wrong, inconsistent, incomplete, ambiguous, or produced under incompatible protocols. It can dominate model behavior when the model is more reliable than the annotation process.

For an observed label $\tilde{y}$ and true latent label $y$, a simple noise model is:

$$
P(\tilde{y}=j\mid y=i)=C_{ij}
$$

where $C$ is a confusion matrix describing how labels are corrupted.

For regression labels:

$$
\tilde{y}=y+\epsilon
$$

where $\epsilon$ may depend on assay, instrument, annotator, batch, or context.

If the irreducible label noise is large, expected performance has a ceiling. A model can be penalized for disagreeing with a noisy observation even when it predicts the latent quantity better:

$$
\mathbb{E}[(\hat{y}-\tilde{y})^2]
=
\mathbb{E}[(\hat{y}-y)^2]
+
\mathbb{E}[\epsilon^2]
$$

when $\epsilon$ is zero-mean and independent of $\hat{y}-y$. The second term is measurement noise, not model error.

## Key Ideas

- Noisy labels can make a good model look wrong or make a memorizing model look strong.
- Noise may be random, systematic, adversarial, batch-specific, or protocol-specific.
- Multiple sources can disagree because they measure different things, not because one is simply wrong.
- In scientific data, label semantics matter: activity, affinity, potency, pose quality, and assay outcome are not interchangeable.
- Missing, weak, and censored labels should be represented explicitly instead of collapsed into ordinary labels.

## Noise Map

| Noise Type | Example | Evaluation Risk |
| --- | --- | --- |
| random | independent annotation mistakes | lowers apparent ceiling |
| systematic | one assay reads high for a target class | model learns protocol artifact |
| source-specific | databases encode different endpoints | harmonization becomes part of the task |
| censoring | value known only above or below a threshold | ordinary regression loss is misleading |
| weak label | proxy label from text, heuristic, or distant supervision | high apparent scale but uncertain semantics |

## Practical Checks

- Who or what produced the label?
- Are repeated measurements consistent?
- Are labels harmonized across protocols, assays, annotators, or benchmarks?
- Is uncertainty or censoring represented explicitly?
- Does the evaluation metric punish predictions that are within label noise?
- Is there an estimate of measurement variability or inter-annotator agreement?
- Are noisy, missing, censored, and weak labels stored as distinct states?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/error-analysis|Error analysis]]
