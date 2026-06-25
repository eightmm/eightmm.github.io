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

## Checks

- What is exactly one example?
- Can multiple rows describe the same underlying entity?
- Does the target label belong to the example, a pair, a group, or a context?
- What metadata is needed to interpret the label?
- Does the split unit need to be broader than the example unit?

## Related

- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/split-unit|Split unit]]
- [[entities/dataset|Dataset]]
- [[entities/entity-relation-map|Entity relation map]]
- [[concepts/evaluation/leakage|Leakage]]
