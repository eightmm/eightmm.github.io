---
title: Entities
tags:
  - bio-ai
  - entities
---

# Entities

Bio-AI에서 먼저 정해야 하는 것은 모델이 다루는 대상입니다. 같은 단어라도 protein, ligand, target, assay, structure가 어떤 단위로 정의되는지에 따라 split, leakage, evaluation이 달라집니다.

$$
x_{\mathrm{bio}}
\in
\{\text{protein}, \text{ligand}, \text{pocket}, \text{complex}, \text{assay}, \text{genome region}\}
$$

## Core Objects

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target|Target]]
- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/ligand|Ligand]]
- [[entities/molecule|Molecule]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[entities/genome|Genome]]

## Label Objects

- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]

## Checks

- Is the entity a biological object, chemical object, assay record, or derived feature?
- Are isoform, construct, mutation, chain, ligand state, and assay context explicit?
- Is the split unit the same as the biological claim?
- Does the input include information unavailable at deployment time?

## Related

- [[bio-ai/index|Bio-AI]]
- [[bio-ai/data-evaluation|Data and evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
