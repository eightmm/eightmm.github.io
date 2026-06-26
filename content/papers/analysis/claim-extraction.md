---
title: Claim Extraction
aliases:
  - papers/claim-extraction
tags:
  - papers
  - methodology
  - workflows
---

# Claim Extraction

Claim extraction turns a paper from a narrative into a set of checkable statements. A paper note should not only say what the paper is about; it should identify what claims are supported, under what protocol, and with what limits.

A useful claim has the form:

$$
\text{method } A
\text{ improves metric } M
\text{ on benchmark } B
\text{ under protocol } P
\text{ compared with baseline } C
$$

## Claim Types

- Method claim: a model, objective, architecture, or data construction is new or useful.
- Empirical claim: a measured result improves over a baseline.
- Mechanism claim: an ablation explains why the method works.
- Generalization claim: the method transfers to new data, tasks, domains, scaffolds, families, or systems.
- Efficiency claim: the method improves memory, latency, throughput, cost, or implementation complexity.
- Scientific claim: the result supports a biological, chemical, or structural interpretation.

## Extraction Steps

1. Identify the paper's central question.
2. List the main claims in one sentence each.
3. Attach each claim to a metric, benchmark, split, baseline, and uncertainty estimate when available.
4. Mark claims that depend on ablations, assumptions, or unverified interpretations.
5. Map the claim into [[papers/analysis/evidence-table|Evidence table]].
6. Move reusable definitions into [[concepts/index|Concepts]].

## Checks

- Is the claim directly supported by a table, figure, experiment, or proof?
- Is the comparison baseline strong and fair?
- Does the split support the generalization claim?
- Is the effect size larger than uncertainty or seed variance?
- Does the paper claim causality from an ablation that is too weak?
- Is the claim still true if phrased narrowly?

## Related

- [[papers/workflows/paper-note-format|Paper note format]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/analysis/evidence-table|Evidence table]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
