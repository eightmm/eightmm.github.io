---
title: Entities
tags:
  - entities
---

# Entities

Entity note는 molecular AI와 protein modeling 전반에 등장하는 object를 정의합니다.

Entity 자체는 research area가 아닙니다. Task, representation, model, label, split, metric이 작동하는 대상입니다.

The basic pattern is:

$$
e
\rightarrow
r=\phi(e,c)
\rightarrow
\hat{y}=f_\theta(r)
$$

where $e$ is an entity, $c$ is context such as target, assay, pocket, source, or time, $\phi$ is a representation function, and $\hat{y}$ is the model output.

Molecular Modeling에서는 같은 entity도 서로 다른 역할을 할 수 있습니다.

| Entity | Common role | Typical risk |
| --- | --- | --- |
| Molecule | input, ligand, generated object | scaffold leakage or inconsistent standardization |
| Protein | target, sequence, structure, context | homolog leakage or missing biological context |
| Pocket | local structure context | ligand-defined pocket leakage |
| Complex | pose, interaction, affinity example | mixing pose quality with binding affinity |
| Assay | measurement context | incompatible labels treated as comparable |
| Dataset | collection and split boundary | row-level split hiding entity overlap |

## Map

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target-assay-label|Target-assay-label contract]]

## Biological and Chemical Objects

- [[entities/target|Target]]
- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/ligand|Ligand]]
- [[entities/molecule|Molecule]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/genome|Genome]]

## Representations and Data

- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[entities/assay|Assay]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/dataset|Dataset]]
- [[entities/target-assay-label|Target-assay-label contract]]

## Checks

- What is one example: molecule, protein, pocket, complex, assay row, sequence region, or dataset record?
- What is the model-ready representation of the entity?
- What context changes the entity's meaning?
- What label, if any, is attached to the entity?
- What split unit prevents the same or nearly same entity from leaking across train and test?
- Does the note describe the object itself, the way to represent it, or the task performed on it?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/data/index|Data]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/evaluation/index|Evaluation]]
