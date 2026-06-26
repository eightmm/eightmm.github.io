---
title: Boltzmann Ceiling Analysis
tags:
  - concept
  - evaluation
  - binding-affinity
---

# Boltzmann Ceiling Analysis

## Definition

Boltzmann ceiling analysis estimates a theoretical upper bound on prediction performance when the task contains examples that are physically or experimentally indistinguishable. In binding-affinity settings, the motivating idea is that mutations with $|\Delta\Delta G| < k_B T$ may be effectively near-neutral under thermal noise and assay uncertainty.

The fraction of near-neutral examples constrains the best possible classifier or ranker:

$$
\text{ceiling}
=
f(p_{\mathrm{neutral}}, \mathcal{D}, m)
$$

where $p_{\mathrm{neutral}}$ is the proportion of examples below the distinguishability threshold, $\mathcal{D}$ is the label distribution, and $m$ is the metric being bounded.

For a thermal-scale rule of thumb:

$$
k_B T \approx 0.593\ \mathrm{kcal/mol}
\quad\text{at}\quad
T = 298\ \mathrm{K}
$$

This does not mean every assay has exactly this noise level. It means that below a relevant physical or experimental distinguishability threshold, ranking or classification labels can become weak evidence.

## Why It Matters

In [[concepts/sbdd/binding-affinity|binding affinity]] prediction, it separates "mutations the physics says are indistinguishable" from "mutations the model fails to learn." This prevents overclaiming model capability and quantifies remaining headroom for improvement.

## Ceiling Inputs

| Input | Meaning |
| --- | --- |
| distinguishability threshold | thermal scale, replicate variance, assay noise, or task-specific minimum effect |
| label distribution | class balance, continuous value density, censoring, and replicate spread |
| metric | accuracy, AUROC, enrichment, Spearman, Kendall, RMSE, or top-k success |
| decision rule | binary threshold, ranking cutoff, selective prediction rule, or regression tolerance |
| subset definition | mutation set, protein family, assay source, ligand series, or benchmark split |

## Interpretation

| Observation | Interpretation |
| --- | --- |
| model far below ceiling | modeling, representation, data, or optimization may be limiting |
| model near ceiling | additional score gains may be hard to interpret without better labels |
| ceiling differs by subset | some targets, assays, or mutation regimes are intrinsically harder |
| ceiling ignores censoring | the bound may be too optimistic |
| ceiling computed after test inspection | it becomes descriptive analysis, not a pre-registered evaluation rule |

## Checks

- What threshold defines near-neutral behavior: $k_B T$, assay noise, replicate variance, or a task-specific bound?
- Is the ceiling derived for a classifier, ranker, or regression metric?
- Does the ceiling account for class imbalance and threshold choice?
- Are censored or weak labels excluded from the ceiling calculation?
- Is the ceiling used as an interpretation aid rather than as a model-quality claim?
- Is the ceiling reported before comparing many model variants, or only after seeing results?
- Are confidence intervals shown for both model performance and ceiling estimates?

## Related Papers

- [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]]

## Related Concepts

- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/data/label-noise|Label noise]]
