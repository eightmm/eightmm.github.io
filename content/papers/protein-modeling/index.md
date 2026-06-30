---
title: Protein Modeling Papers
unlisted: true
tags:
  - papers
  - protein-modeling
---

# Protein Modeling Papers

Protein modeling paper note는 structure prediction, sequence model, protein representation learning, MSA usage, sequence-structure evaluation을 다룹니다.

이 선반은 protein 자체가 중심 object일 때 사용합니다. Protein-ligand docking, pocket scoring, virtual screening이 중심이면 [[papers/sbdd/index|Structure-Based Modeling Papers]]로 두고 여기에는 cross-link합니다.

$$
\hat{z}, \hat{X}, \hat{y}
=
f_\theta(s, X, c)
$$

where $s$ is sequence, $X$ is optional structure, and $c$ may include MSA, template, family, domain, assay, peptide, antibody, or interaction context.

## Reading Axes

- What is the input: sequence, MSA, template, structure, or mixed context?
- What is predicted: coordinates, contact map, representation, function, or confidence?
- What architecture is used: Transformer, state-space model, equivariant model, or hybrid?
- How are homolog leakage, template leakage, and protein-family splits handled?
- Is confidence, uncertainty, or failure mode analysis reported?

## Routing

| Strongest Claim | Put Here? | Cross-Link |
| --- | --- | --- |
| protein sequence representation or pLM | yes | [Protein modeling concepts](/concepts/protein-modeling) |
| structure prediction or contact/distance modeling | yes | [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction) |
| antibody, peptide, or protein-protein interaction | yes, if protein object is central | [Interaction modeling](/molecular-modeling/interactions) |
| protein-ligand pose, docking, scoring, screening | usually no | [SBDD papers](/papers/sbdd) |
| molecule generation with protein conditioning | usually no | [Generative model papers](/papers/generative-models), [Computational Biology papers](/papers/computational-biology) |
| architecture block evaluated on proteins | depends | [Architecture papers](/papers/architectures) |

## Evidence Boundary

Protein papers often mix sequence, structure, and function claims. Read the evidence by type.

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| sequence likelihood or masked-token accuracy | sequence distribution and representation learning | function, binding, expression, stability |
| protein-family split benchmark | transfer beyond close homologs | deployment to new assay or chemical context |
| contact/distance accuracy | coarse geometry | ligand-ready pocket or side-chain quality |
| backbone/all-atom structure metric | structure prediction quality | docking, affinity, or functional activity |
| confidence calibration | reliability of predicted error | biological correctness |
| downstream task score | task-specific utility | broad protein understanding |

## Paper Note Fields

| Field | Ask |
| --- | --- |
| Protein unit | residue, domain, chain, family, pocket, complex, antibody, peptide? |
| Sequence source | database, filtering, redundancy, family split, date? |
| Structure source | experimental, predicted, template-derived, relaxed, apo/holo? |
| Context | MSA, template, ligand, assay, species, construct, antibody pair? |
| Output | embedding, contact, coordinate, confidence, function, binding, design? |
| Metric | sequence, structure, ranking, interaction, or function metric? |
| Leakage | homolog, template, structure, partner, assay/source, or time leakage? |

## Concepts

- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/protein-modeling/protein-language-model|Protein language model]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]

## Curated Notes

- [[papers/architectures/alphafold2|AlphaFold2]]
- [[papers/protein-modeling/meet-equivariant-peptide|MEET]]
- [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]]

## Related

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[papers/analysis/claim-extraction|Claim extraction]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
