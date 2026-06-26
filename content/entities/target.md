---
title: Target
tags:
  - entities
  - molecular-modeling
---

# Target

A target is the biological object a model or experiment is centered on. In this wiki, a target is usually a [[entities/protein|protein]], [[entities/pocket|pocket]], [[entities/sequence|sequence]], or genomic region rather than a broad disease program.

A task can be written as:

$$
\hat{y} = f_\theta(x_{\mathrm{molecule}}, x_{\mathrm{target}}, c)
$$

where $x_{\mathrm{molecule}}$ is a molecule or ligand representation, $x_{\mathrm{target}}$ is target context, and $c$ is assay or experimental context.

When labels are measured, the target should be interpreted through the [[entities/target-assay-label|Target-assay-label contract]] rather than as a standalone identifier.

## Target Context

- Protein sequence.
- Protein family or domain.
- 3D structure or predicted structure.
- Binding pocket.
- Species, isoform, mutation, or construct when public and relevant.
- Assay context when labels come from experiments.

## Target Identity Contract

| Field | Question | Why It Matters |
| --- | --- | --- |
| public identifier | which public target entry is used? | reproducibility |
| sequence | which amino acid or genomic sequence? | homolog split and mutation effects |
| family/domain | which family or domain grouping? | transfer claim boundary |
| structure | apo, holo, predicted, experimental, relaxed? | docking and structure claims |
| pocket | global target or local binding site? | interaction and pose context |
| species/isoform | which public biological variant? | label comparability |
| assay context | which measurement setup? | target label is not context-free |

For target-conditioned prediction:

$$
p_\theta(y\mid m,t,a)
$$

is not the same task as molecule-only prediction:

$$
p_\theta(y\mid m)
$$

If a paper describes target-aware modeling, the target representation and target split must be explicit.

## Target Split Boundary

| Claim | Required Split |
| --- | --- |
| new molecules for known targets | molecule/scaffold split |
| new targets in known family | target-held-out split with family analysis |
| new protein families | protein-family split |
| new pockets or structures | pocket/structure-source split |
| new target-assay settings | target plus assay/source split |

Target-level generalization cannot be proven by a row-random split.

## Structure-Based Context

When a target is used as a structure:

$$
t \rightarrow (s, X, P)
$$

where $s$ is sequence, $X$ is coordinates, and $P$ is pocket definition. The note should say whether $X$ is experimental, predicted, template-derived, apo, holo, or docked.

## Checks

- Is the target a protein, pocket, family, sequence region, or assay-defined object?
- Are target identifiers public and reproducible?
- Are isoform, mutation, and construct differences treated explicitly?
- Are target-level splits separated from molecule-level splits?
- Is the target tied to an assay context, label unit, and split rule?
- Is private collaborator or project context removed?
- Is the target representation available at inference time?
- Are target, pocket, and protein-ligand-complex claims kept separate?

## Related

- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[molecular-modeling/proteins|Proteins]]
