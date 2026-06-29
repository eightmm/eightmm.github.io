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

The prediction target must say what "effect" means:

$$
\text{effect}
=
\text{change in a specified observable under a specified context}
$$

Without that context, a variant effect score is only a generic ranking signal.

## Task Contract

| Field | Question |
| --- | --- |
| Variant unit | SNV, insertion/deletion, haplotype, edited region, or window |
| Reference context | assembly, strand, window length, flanking sequence |
| Alternate context | how the alternate sequence is constructed |
| Target observable | annotation, assay readout, expression proxy, constraint score, or benchmark label |
| Label source | measured, curated, simulated, predicted, or derived |
| Split unit | region, locus, chromosome, source, variant family, or species |
| Metric | classification, regression, ranking, calibration, or retrieval metric |

## Reference and Alternate Windows

For a fixed window operator $W$, a common construction is:

$$
x_{\mathrm{ref}} = W(G, r),
\qquad
x_{\mathrm{alt}} = \operatorname{edit}(x_{\mathrm{ref}}, v)
$$

where $G$ is the reference genome and $r$ is the region around the variant. The edit operation should be deterministic and documented.

| Choice | Risk |
| --- | --- |
| window length | too short loses regulatory or protein-context information |
| strand handling | reverse-complement mismatch |
| indel handling | coordinate shift and alignment ambiguity |
| multiple variants | single-variant assumption breaks |
| assembly version | ref/alt sequence mismatch |

## Label Types

| Label Type | Meaning | Caution |
| --- | --- | --- |
| functional assay readout | measured change in a defined experiment | assay/source-specific |
| computational annotation | label produced by another model or rule | teacher leakage and circular evidence |
| conservation or constraint proxy | evolutionary signal | not the same as experimental function |
| public benchmark label | task-specific evaluation label | benchmark scope may be narrow |
| clinical category | interpretation label | out of scope here unless explicitly public and non-sensitive |

## Evaluation Boundary

Variant effect prediction can be evaluated as classification, regression, ranking, or prioritization. The metric should match the output claim.

| Claim | Evidence Needed |
| --- | --- |
| variant classification | label semantics, threshold, calibration or confusion matrix |
| effect-size regression | unit, transform, uncertainty, error scale |
| prioritization | ranked-list metric and candidate set definition |
| cross-region generalization | region/locus split and overlap checks |
| cross-source robustness | source-held-out or annotation-version-held-out evaluation |

For a score difference $\Delta_\theta(v)$, calibration is not automatic. A larger score may rank variants without being a well-calibrated probability or measured effect size.

## Leakage Risks

| Risk | Check |
| --- | --- |
| nearby variants in train and test | group by region or locus |
| overlapping windows | split before window extraction |
| same source evidence in labels | record source and version |
| annotation-derived labels used as features | separate input features from targets |
| homologous or duplicated regions | group duplicated sequence contexts |
| post-hoc threshold tuning | tune threshold on validation only |

## Checks

- What target does "effect" mean?
- Are reference and alternate contexts generated from the same assembly?
- Are variants near each other split carefully to avoid regional leakage?
- Are labels measured, curated, simulated, or inferred?
- Is the model evaluated on held-out regions, variant types, or sources?
- Is the task classification, regression, ranking, or prioritization?
- Does the metric match the claim rather than only the model output format?
- Are threshold choices, candidate sets, and failed variants included in the protocol?
- Is the note staying within sequence/region/variant modeling rather than clinical interpretation?

## Related

- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[entities/genome|Genome]]
- [[entities/sequence|Sequence]]
- [[entities/dataset|Dataset]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
