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

## Key Ideas

- Noisy labels can make a good model look wrong or make a memorizing model look strong.
- Noise may be random, systematic, adversarial, batch-specific, or protocol-specific.
- Multiple sources can disagree because they measure different things, not because one is simply wrong.
- In scientific data, label semantics matter: activity, affinity, potency, pose quality, and assay outcome are not interchangeable.

## Practical Checks

- Who or what produced the label?
- Are repeated measurements consistent?
- Are labels harmonized across protocols, assays, annotators, or benchmarks?
- Is uncertainty or censoring represented explicitly?
- Does the evaluation metric punish predictions that are within label noise?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/error-analysis|Error analysis]]
