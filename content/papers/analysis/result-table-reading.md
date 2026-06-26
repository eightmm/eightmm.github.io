---
title: Result Table Reading
tags:
  - papers
  - evaluation
  - methodology
---

# Result Table Reading

Result table reading is the process of converting a paper table into narrow claims. A table entry is not a conclusion by itself; it is a metric under a dataset, split, protocol, selection rule, and uncertainty boundary.

The compact form is:

$$
(\text{row},\ \text{column},\ \text{metric},\ \text{protocol})
\rightarrow
\text{claim}
$$

## Parse Order

| Step | Question |
| --- | --- |
| Task | What input-output problem does the table evaluate? |
| Dataset | Which benchmark, split, filtering, and preprocessing define the examples? |
| Metric | Is higher or lower better, and what is being averaged? |
| Row | Does the row change architecture, objective, data, compute, or postprocessing? |
| Column | Does the column change dataset, metric, subset, or evaluation condition? |
| Selection | Is the value best checkpoint, best seed, mean, median, or single run? |
| Uncertainty | Are confidence interval, seed variance, paired difference, or test-set size reported? |

## Claim Types

| Table Pattern | Safer Claim |
| --- | --- |
| One bold number | model is best on that metric/protocol, not globally best |
| Mean ± std | effect should be compared to run-to-run variance |
| Multiple datasets | inspect whether gains are consistent or driven by one dataset |
| Ablation table | check whether only one factor changed |
| Leaderboard table | verify train/test boundary and repeated submission risk |
| Molecular benchmark | check scaffold, protein-family, assay, and template leakage |

## Difference Reading

For two rows $A$ and $B$:

$$
\Delta M
=
M_A-M_B
$$

The important object is not only $\Delta M$, but whether it is interpretable:

$$
\text{interpretable improvement}
=
\Delta M
\land
\text{same protocol}
\land
\text{relevant baseline}
\land
\text{uncertainty boundary}
$$

## Red Flags

- Bold value is a single run while competitors report means.
- The table mixes models with different data, pretraining, compute, or postprocessing.
- Standard deviation is across test examples, but the text treats it as seed variance.
- A validation-selected threshold is evaluated repeatedly on the test set.
- A small improvement is highlighted without paired comparison or confidence interval.
- Molecular results ignore scaffold split, target split, assay source, or pose validity.

## Related

- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
