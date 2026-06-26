---
title: Paper Analysis
unlisted: true
tags:
  - papers
  - methodology
  - evaluation
---

# Paper Analysis

Paper analysis notes extract claims, evidence, benchmarks, ablations, limitations, and comparison axes from papers.

The analysis layer turns prose into checkable structure:

$$
\text{claim}
\rightarrow
\text{evidence}
\rightarrow
\text{scope}
\rightarrow
\text{limitation}
$$

This keeps a paper note from becoming only a summary of the abstract.

## Scope

- Claim extraction and evidence tables.
- Benchmark cards, split interpretation, and metric selection.
- Ablation maps for component claims.
- Limitation taxonomy and paper comparison matrices.

## Notes

- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]

## Checks

- Does each important claim name method, task, benchmark, metric, protocol, and baseline?
- Does each evidence item state the scope it actually supports?
- Are ablations tied to components rather than treated as generic performance numbers?
- Are limitations classified as data, split, metric, baseline, ablation, generalization, reproducibility, efficiency, or domain limits?
- Are cross-paper comparisons made on the same task and evaluation boundary?

## Where New Notes Go

- Analysis templates and claim structures go here.
- Reproduction planning goes under [[papers/reproducibility/index|Paper reproducibility]].
- General evaluation concepts go under [[concepts/evaluation/index|Evaluation]].
- Research-method claims for your own work go under [[concepts/research-methodology/index|Research methodology]].

## Related

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/data/benchmark|Benchmark]]
- [[papers/workflows/index|Paper workflows]]
- [[papers/reproducibility/index|Paper reproducibility]]
