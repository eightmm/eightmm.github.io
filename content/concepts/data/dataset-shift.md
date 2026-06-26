---
title: Dataset Shift
tags:
  - data
  - evaluation
  - generalization
---

# Dataset Shift

Dataset shift occurs when the training distribution differs from validation, test, or deployment distributions. It is one of the main reasons a model can look good in a benchmark and fail in real use.

The general problem is:

$$
p_{\mathrm{train}}(x,y)
\ne
p_{\mathrm{test}}(x,y)
$$

or more importantly:

$$
p_{\mathrm{train}}(x,y)
\ne
p_{\mathrm{deploy}}(x,y)
$$

## Common Types

Covariate shift changes the input distribution:

$$
p_{\mathrm{train}}(x)
\ne
p_{\mathrm{deploy}}(x),
\qquad
p(y\mid x)\ \text{approximately stable}
$$

Label shift changes class or target frequencies:

$$
p_{\mathrm{train}}(y)
\ne
p_{\mathrm{deploy}}(y),
\qquad
p(x\mid y)\ \text{approximately stable}
$$

Concept shift changes the input-output relationship:

$$
p_{\mathrm{train}}(y\mid x)
\ne
p_{\mathrm{deploy}}(y\mid x)
$$

Concept shift is often the hardest because the meaning of the target changes.

## Sources

- Different data collection protocol.
- Different time period, site, instrument, species, target, or user population.
- Different preprocessing pipeline.
- Different annotation policy.
- Benchmark construction artifacts.
- Deployment inputs outside the training support.

## Shift Contract

| Field | Question |
| --- | --- |
| Source distribution | What data generated training examples? |
| Target distribution | What data should the claim generalize to? |
| Shift axis | molecule, target, assay, time, source, structure, modality, or user context? |
| Split simulation | Which split approximates that shift? |
| Metric | Which metric is reported under shift? |
| Slice size | Is each shifted slice large enough? |
| Applicability domain | Where should the model not be trusted? |

The OOD gap should be tied to a named shift:

$$
\Delta_{\mathrm{shift}}
=
M_{\mathrm{in}} - M_{\mathrm{shift}}
$$

for higher-is-better metrics, or the sign should be reversed for loss-like metrics.

## Chem-Bio Shift Axes

| Axis | Example | Better Evaluation |
| --- | --- | --- |
| molecule scaffold | new chemotypes | scaffold split, activity-cliff analysis |
| target family | new protein family | protein-family split |
| target-assay pair | new measurement context | target-assay held-out split |
| assay source | new public database/source | source-held-out split |
| structure source | predicted vs experimental structures | structure-source slice |
| time | later database release | temporal split |
| label policy | threshold or unit change | label semantics audit |

For structure-based work, shift can also come from receptor preparation, pocket definition, conformer generation, or docking failure policy.

## Shift vs Leakage

Shift and leakage are different failure modes:

| Problem | Symptom |
| --- | --- |
| shift | test distribution is legitimately harder or different |
| leakage | test examples contain information already available during training |
| benchmark artifact | shortcut correlates with label in both train and test |
| label mismatch | target meaning changes across sources |

A benchmark can have both shift and leakage. Report both checks separately.

## Checks

- Which distribution does each split represent?
- Is the shift intentional, such as scaffold split or family split?
- Is the shift visible in metadata?
- Are performance drops reported by source, subgroup, time, or entity?
- Does the metric hide poor behavior on shifted subsets?
- Is the shifted test set used for model selection?
- Does preprocessing depend on statistics from the shifted test set?
- Are shifted-slice confidence intervals or sample counts reported?

## Related

- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
