---
title: Paper Comparison Matrix
unlisted: true
aliases:
  - papers/paper-comparison-matrix
tags:
  - papers
  - methodology
  - workflows
---

# Paper Comparison Matrix

A paper comparison matrix compares several papers along the same axes. It is useful when a topic has many related methods and a prose summary would hide important differences.

## Suggested Columns

| Axis | Meaning |
| --- | --- |
| Problem | What task or question is addressed? |
| Input | What modality or entity is used? |
| Output | What does the model predict or generate? |
| Architecture | What model family is used? |
| Objective | What loss or training signal is optimized? |
| Data | What dataset, benchmark, and preprocessing are used? |
| Split | What generalization boundary is tested? |
| Metric | What number supports the claim? |
| Baseline | What comparison makes the result meaningful? |
| Ablation | What explains the gain? |
| Evidence | Which claim is supported by which experiment? |
| Limit | What remains uncertain or weak? |
| Reproducibility | What public artifacts or details are available? |
| Wiki links | Which reusable concepts should be updated? |

## When To Use

- Several papers solve the same task with different architectures.
- One method family appears across molecules, proteins, text, images, or agents.
- Benchmarks differ enough that leaderboard-style comparison is misleading.
- A blog post needs a compact map before a deeper explanation.

## Checks

- Are all rows using comparable metrics and splits?
- Are missing values marked explicitly instead of guessed?
- Are papers grouped by task rather than by popularity?
- Does the matrix reveal a real distinction, or only restate abstracts?
- Which concept notes should be updated after the comparison?

## Related

- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/metric|Metric]]
