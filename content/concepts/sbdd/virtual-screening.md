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

## Checks

- Is the library filtered for chemistry, duplicates, and assay compatibility?
- Are decoys realistic, or do they make the task artificially easy?
- Is ranking driven by pose quality, affinity, ligand bias, or dataset artifacts?
- Are top-ranked molecules inspected for validity and novelty?
- Are [[concepts/evaluation/negative-set|negative sets]] and [[concepts/evaluation/applicability-domain|applicability domain]] reported?
- Is pose generation failure separated from scoring failure?
- Is early enrichment reported instead of only global classification metrics?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/metric|Metric]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
