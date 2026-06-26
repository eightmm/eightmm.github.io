---
title: Claim-Evidence Boundary
tags:
  - evaluation
  - methodology
  - claims
---

# Claim-Evidence Boundary

A claim-evidence boundary states exactly what a result proves and what it does not prove. It is useful for papers, Korean synthesis posts, project notes, and concept pages.

The minimal form is:

$$
(\text{claim},\ \text{task},\ \text{data},\ \text{protocol},\ \text{metric},\ \text{baseline},\ \text{uncertainty})
\rightarrow
\text{supported scope}
$$

If any part is missing, the claim should be narrowed or marked `to verify`.

## Boundary Fields

| Field | Question |
| --- | --- |
| Claim | What exact statement is being made? |
| Task | What input and output space does the evidence test? |
| Data | Which population, benchmark, preprocessing, and split define the examples? |
| Protocol | How were model selection, threshold choice, early stopping, and final testing separated? |
| Metric | What number supports the claim, and what does it average over? |
| Baseline | What comparison makes the result meaningful? |
| Uncertainty | Is the effect larger than seed variation, confidence interval, or measurement noise? |
| Scope | Where should the claim be expected to hold? |

## Narrowing Rule

Write the broad version first, then narrow it until every word is supported.

| Broad Claim | Narrower Claim |
| --- | --- |
| This model generalizes to proteins. | This model improves metric $M$ on a protein-family split for benchmark $B$. |
| This method improves docking. | This method improves pose plausibility under the reported pocket and ligand-preparation protocol. |
| This architecture is better. | This architecture improves metric $M$ under matched data, objective, and training budget. |
| This objective learns better representations. | This objective improves downstream metric $M$ under the stated frozen or fine-tuned protocol. |
| This benchmark proves real-world performance. | This benchmark supports the specific population, split, and allowed-information boundary it defines. |

## Formula View

An empirical claim usually estimates a target quantity:

$$
Q
=
\mathbb{E}_{z\sim p_{\mathrm{target}}}
\left[
m(f_\theta, z)
\right]
$$

but reports a finite estimate:

$$
\hat{Q}
=
\frac{1}{n}
\sum_{i=1}^{n}
m(f_\theta, z_i),
\qquad
z_i \sim \hat{p}_{\mathrm{eval}}
$$

- $Q$: target quantity the claim wants to support.
- $\hat{Q}$: measured estimate.
- $z$: evaluation unit, such as example, query, molecule, complex, pose, target, or task.
- $p_{\mathrm{target}}$: intended population or deployment distribution.
- $\hat{p}_{\mathrm{eval}}$: benchmark or test-set distribution actually sampled.
- $m$: metric contribution for one unit.

The claim is only as broad as the match between $p_{\mathrm{target}}$ and $\hat{p}_{\mathrm{eval}}$.

## Molecular Modeling Checks

| Claim Type | Boundary To State |
| --- | --- |
| Molecule property | molecule standardization, label source, scaffold or time split, metric |
| Protein prediction | sequence identity or family split, residue indexing, structure source, metric |
| Protein-ligand complex | ligand scaffold, protein family, complex-pair split, pocket definition, coordinate source |
| Docking pose | receptor preparation, ligand state, pose RMSD rule, geometry checks, known-pocket assumption |
| Virtual screening | active/decoy construction, enrichment metric, property bias, target context |
| Generative molecule | validity, uniqueness, novelty, diversity, constraint satisfaction, filtering rule |

## Use In Posts

For Korean synthesis posts, include only the boundary needed for the reader:

- What claim is being discussed?
- What evidence supports it?
- What split or benchmark limits the claim?
- Which wiki note holds the detailed metric, formula, or paper evidence?

Long evidence tables belong in [[papers/analysis/evidence-table|Evidence table]] or paper notes. The post should keep the boundary readable.

## Red Flags

- A random split is used to claim new-scaffold or new-protein generalization.
- A metric is reported without model-selection or threshold-selection details.
- A benchmark score is treated as deployment performance without population matching.
- A molecular result ignores standardization, assay context, or leakage.
- A small score difference is interpreted without uncertainty or paired comparison.
- A qualitative example is used to imply a broad quantitative claim.

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/data/benchmark-intake|Benchmark intake]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/leakage|Leakage]]
- [[bio/data-evaluation|Molecular modeling data and evaluation]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/evidence-table|Evidence table]]
