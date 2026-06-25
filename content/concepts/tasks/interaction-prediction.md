---
title: Interaction Prediction
tags:
  - tasks
  - protein-ligand
  - bio-ai
---

# Interaction Prediction

Interaction prediction estimates whether two or more entities interact, how strongly they interact, or what structured relationship they form. In Bio-AI this often means molecule-target, protein-ligand, protein-protein, residue-residue, or molecule-molecule interactions.

A pairwise interaction model can be written as:

$$
\hat{y}
=
f_\theta(r_a, r_b, c)
$$

where $r_a$ and $r_b$ are entity representations and $c$ is context such as assay, pocket, species, source, or geometric frame.

For a protein-ligand complex:

$$
\hat{y}
=
f_\theta(r_P, r_L, r_{PL})
$$

where $r_P$ is protein or pocket representation, $r_L$ is ligand representation, and $r_{PL}$ is an optional pair or interaction representation.

## Output Variants

- Binary interaction: interact vs. not interact.
- Affinity or activity: scalar label such as binding affinity or assay response.
- Contact map: pairwise interaction matrix.
- Pose or geometry: structured coordinates or relative transform.
- Ranking score: prioritize candidates for screening.

For pairwise contacts:

$$
\hat{C}_{ij}
=
\sigma(s_\theta(h_i^{(a)}, h_j^{(b)}, e_{ij}))
$$

where $h_i^{(a)}$ and $h_j^{(b)}$ are entity features, $e_{ij}$ is pair evidence such as distance or edge features, and $\hat{C}_{ij}$ is an interaction probability.

## Evaluation Risks

- Random row splits can leak the same entity pair, close ligand scaffold, homologous protein, or shared assay context.
- Negative labels may mean measured non-interaction, unobserved interaction, or untested pair.
- A model can learn assay/source identity rather than interaction mechanism.
- Pose-based features may not be available at deployment time.
- Binding affinity, pose quality, and interaction existence are different targets.

## Checks

- What are the interacting entities?
- Is the task binary prediction, affinity prediction, pose prediction, contact prediction, or ranking?
- Is the label measured, curated, inferred, censored, weak, or missing?
- What pair, scaffold, family, assay, source, or temporal split is required?
- Does the representation include only information available at prediction time?
- Are interaction evidence, label semantics, and metric aligned?

## Related

- [[concepts/tasks/property-prediction|Property prediction]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
