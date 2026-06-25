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

## Checks

- Which distribution does each split represent?
- Is the shift intentional, such as scaffold split or family split?
- Is the shift visible in metadata?
- Are performance drops reported by source, subgroup, time, or entity?
- Does the metric hide poor behavior on shifted subsets?

## Related

- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
