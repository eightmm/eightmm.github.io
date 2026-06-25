---
title: Example Unit
tags:
  - data
  - machine-learning
---

# Example Unit

An example unit defines what one row, item, case, or training example represents. It is the first data question because labels, splits, metrics, and leakage all depend on the unit.

A supervised dataset can be written as:

$$
\mathcal{D}
=
\{(u_i, x_i, y_i, m_i)\}_{i=1}^{n}
$$

$u_i$ is the example unit identifier, $x_i$ is input, $y_i$ is the target, and $m_i$ is metadata.

## Common Units

- Image or video clip.
- Text document, passage, prompt, or conversation.
- Molecule, ligand, protein, complex, assay row, or dataset record.
- Sequence, genomic region, variant, or structure.
- User action, event, request, job, or run.

## Why It Matters

Changing the example unit changes the task. A molecule-level prediction, molecule-target prediction, and assay-measurement prediction can use similar features but answer different questions.

For pair or context-dependent tasks, the unit may be a tuple:

$$
u_i = (e_i, c_i)
$$

where $e_i$ is the main entity and $c_i$ is context such as target, assay, prompt, pocket, source, or time. The same entity can appear in multiple examples if the context changes.

## Ambiguous Cases

- Molecule property: unit may be molecule, assay row, molecule-target pair, or measurement.
- Protein task: unit may be sequence, domain, family, structure, residue, or complex.
- Docking task: unit may be receptor-ligand pair, pose, pocket-ligand pair, or screening candidate.
- LLM task: unit may be prompt, message, conversation, document chunk, or user request.

## Checks

- What is exactly one example?
- Can multiple rows describe the same underlying entity?
- Does the target label belong to the example, a pair, a group, or a context?
- What metadata is needed to interpret the label?
- Does the split unit need to be broader than the example unit?
- Would changing the unit change the loss, metric, or leakage risk?

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/split-unit|Split unit]]
- [[entities/dataset|Dataset]]
- [[entities/entity-relation-map|Entity relation map]]
- [[concepts/evaluation/leakage|Leakage]]
