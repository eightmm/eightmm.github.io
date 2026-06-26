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

## Checks

- Is the target a protein, pocket, family, sequence region, or assay-defined object?
- Are target identifiers public and reproducible?
- Are isoform, mutation, and construct differences treated explicitly?
- Are target-level splits separated from molecule-level splits?
- Is the target tied to an assay context, label unit, and split rule?
- Is private collaborator or project context removed?

## Related

- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
