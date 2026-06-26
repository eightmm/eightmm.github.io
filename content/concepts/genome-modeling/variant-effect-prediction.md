---
title: Variant Effect Prediction
tags:
  - genome
  - molecular-modeling
  - evaluation
---

# Variant Effect Prediction

Variant effect prediction estimates how a sequence variant changes a biological or functional target. In this wiki, it is treated as a genome-level sequence modeling task, not as a clinical interpretation workflow.

A variant can be represented as:

$$
v = (r, b_{\mathrm{ref}}, b_{\mathrm{alt}})
$$

where $r$ is the genomic position or region, $b_{\mathrm{ref}}$ is the reference allele, and $b_{\mathrm{alt}}$ is the alternate allele.

A model often compares reference and alternate sequence contexts:

$$
\Delta_\theta(v)
= f_\theta(x_{\mathrm{alt}}) - f_\theta(x_{\mathrm{ref}})
$$

## Label Types

- Functional assay readout.
- Computational annotation.
- Conservation or constraint proxy.
- Public benchmark label.

## Checks

- What target does "effect" mean?
- Are reference and alternate contexts generated from the same assembly?
- Are variants near each other split carefully to avoid regional leakage?
- Are labels measured, curated, simulated, or inferred?
- Is the model evaluated on held-out regions, variant types, or sources?

## Related

- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
