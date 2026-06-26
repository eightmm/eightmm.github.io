---
title: Virtual Screening
tags:
  - sbdd
  - docking
  - drug-discovery
---

# Virtual Screening

Virtual screening ranks a library of candidate molecules for a target. In structure-based workflows, ranking may use docking poses, scoring functions, learned affinity models, or filters.

Given a target pocket $P$ and a molecule library $\mathcal{L}$, a simple screening objective is:

$$
\operatorname{rank}(L)
= S(P,L,\hat{X}_L)
$$

where $\hat{X}_L$ is the generated or selected pose for ligand $L$, and $S$ is a scoring function.

Enrichment at top $k$ can be written as:

$$
\operatorname{EF}_k
=
\frac{\text{actives in top }k / k}
{\text{actives in library} / |\mathcal{L}|}
$$

## Screening Contract

Virtual screening is a ranking task over a specific candidate pool:

$$
\mathcal{C}
=
\{(L_i, r_i, a_i)\}_{i=1}^{N}
$$

where $L_i$ is a ligand candidate, $r_i$ is its prepared representation, and $a_i$ is activity or relevance metadata when available.

For a reusable note, state:

| Field | Question |
| --- | --- |
| target context | protein, pocket, receptor state, cofactors, waters, and assay context |
| library source | known actives, measured inactives, decoys, purchasable library, generated set, or public benchmark |
| ligand preparation | standardization, protonation, tautomer, stereochemistry, conformers |
| pose policy | docked pose, known pose, generated pose, ensemble, or no pose |
| scoring policy | docking score, learned score, consensus, filter, or reranker |
| denominator | all attempted compounds, successfully docked compounds, or post-filtered compounds |

The denominator is part of the claim. Reporting only post-filtered molecules can overstate practical screening quality.

## Ranking Pipeline

A structure-based screening workflow is often:

$$
L
\rightarrow
\tilde{L}
\rightarrow
\{\hat{X}_{L,j}\}_{j=1}^{J}
\rightarrow
s(L)
\rightarrow
\operatorname{rank}(L)
$$

Each arrow can fail. A paper should separate ligand preparation failure, pose generation failure, scoring failure, and final ranking quality.

## Metrics

Use ranking metrics that match the screen:

| Metric | Use | Main Caveat |
| --- | --- | --- |
| EF@K or EF at top fraction | early active enrichment | sensitive to active prevalence and decoys |
| ROC-AUC | global ranking over positives/negatives | can look good while early enrichment is weak |
| BEDROC or early-recognition metric | top-heavy ranking | parameter choice affects interpretation |
| precision@K | fixed wet-lab or compute budget | depends on candidate pool and label completeness |
| hit rate after filters | practical pipeline yield | denominator must include failed candidates |

For docking-based screens, pose plausibility and enrichment should be reported separately.

## Checks

- Is the library filtered for chemistry, duplicates, and assay compatibility?
- Are decoys realistic, or do they make the task artificially easy?
- Is ranking driven by pose quality, affinity, ligand bias, or dataset artifacts?
- Are top-ranked molecules inspected for validity and novelty?
- Are [[concepts/evaluation/negative-set|negative sets]] and [[concepts/evaluation/applicability-domain|applicability domain]] reported?
- Is pose generation failure separated from scoring failure?
- Is early enrichment reported instead of only global classification metrics?
- Are failed, invalid, or filtered molecules included in sample accounting?
- Is the target split new protein, new ligand scaffold, new pair, or within-family interpolation?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/metric|Metric]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
