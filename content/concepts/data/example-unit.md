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

| Domain | Possible Example Unit | Typical Context |
| --- | --- | --- |
| Vision | image, video clip, frame, object instance | camera, scene, timestamp, annotation source |
| Language | document, passage, prompt, message, conversation | source, user request, retrieval context |
| Molecules | standardized molecule, conformer, ligand, assay row | target, assay, endpoint, chemical state |
| Proteins | sequence, residue, domain, structure, protein family | species, construct, structure source, MSA context |
| Structure-based modeling | receptor-ligand pair, pocket-ligand pair, pose, complex | receptor state, pocket definition, ligand pose source |
| Genome sequence | sequence window, genomic region, variant | reference genome, strand, coordinate system, cell or tissue context when public |
| Systems | request, job, run, event, user action | environment, version, time block, policy |

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

## Unit Changes That Change the Claim

| If the unit is | The model claims | Main leakage risk |
| --- | --- | --- |
| molecule | a property attached to a chemical identity | duplicate salts, tautomers, stereo variants, scaffold overlap |
| molecule-target pair | target-conditioned activity or affinity | same molecule or same target appearing across splits |
| assay measurement | a protocol-dependent observed value | assay, source, threshold, or campaign leakage |
| protein sequence | sequence-level behavior | close homologs across train and test |
| residue | local annotation or residue-level output | same protein split into train and test residues |
| protein-ligand pose | pose scoring or pose classification | generated poses from the same complex crossing splits |
| genomic region | region-level prediction | adjacent windows or same locus crossing splits |

The unit should be chosen before model design. Otherwise the same score can be read as a stronger generalization claim than the dataset supports.

## Minimal Contract

For a reusable note, state:

- Unit identifier: what makes two examples the same or different.
- Input fields: raw object, context object, metadata, and required preprocessing.
- Label owner: whether the label belongs to the entity, pair, assay, pose, region, or run.
- Split implication: which broader group must stay together.
- Metric implication: whether metrics aggregate by row, entity, target, scaffold, family, or source.

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
