---
title: Computational Biology Papers
tags:
  - papers
  - computational-biology
---

# Computational Biology Papers

Computational Biology paper note는 structure-based modeling, protein modeling, molecular generation, protein-ligand interaction, assay-aware label, domain-specific evaluation을 다룹니다.

이곳은 computational-biology paper를 읽기 위한 큰 선반입니다. [[papers/sbdd/index|Structure-based modeling papers]]와 [[papers/protein-modeling/index|Protein modeling papers]] 같은 좁은 묶음은 세부 주제가 필요할 때만 따라갑니다.

## Reading Axes

- 어떤 object를 모델링하는가: protein, molecule, ligand, pocket, complex, pose, assay record, sequence region?
- 중심 route가 무엇인가: [[molecular-modeling/sequence-based|sequence-based modeling]], [[molecular-modeling/molecular-ligand|molecular and ligand modeling]], [[molecular-modeling/interactions|interaction modeling]], [[molecular-modeling/structure-based/index|structure-based modeling]]?
- contribution이 AI method, domain workflow, benchmark, dataset, evaluation protocol 중 무엇인가?
- molecule standardization, protein-family split, scaffold split, assay semantics, leakage risk가 explicit한가?
- paper가 method quality를 benchmark artifact나 domain shortcut과 분리하는가?

## Sub-Buckets

| Bucket | Use for | Notes |
| --- | --- | --- |
| [Structure-based modeling papers](/papers/sbdd) | docking, pose quality, scoring, conformers, virtual screening | [PoseBusters](/papers/sbdd/posebusters) |
| [Protein modeling papers](/papers/protein-modeling) | protein representation, structure, antibody/protein interaction | [MEET](/papers/protein-modeling/meet-equivariant-peptide), [Multi-scale antibody binding](/papers/protein-modeling/multi-scale-antibody-binding) |
| [Generative model papers](/papers/generative-models) | molecule/protein generation when generation is the strongest claim | [Molexar](/papers/generative-models/molexar) |

## Active Notes

| Paper note | Main route | 여기 두는 이유 |
| --- | --- | --- |
| [PoseBusters](/papers/sbdd/posebusters) | structure-based evaluation | pose plausibility and docking evaluation |
| [AlphaFold2](/papers/architectures/alphafold2) | protein structure prediction | MSA-pair architecture, geometric structure module, recycling, and confidence |
| [AlphaFold3](/papers/architectures/alphafold3) | biomolecular complex prediction | diffusion-based joint structure prediction for proteins, nucleic acids, ligands, ions, and modified residues |
| [ProteinMPNN](/papers/architectures/proteinmpnn) | protein inverse folding | structure-conditioned sequence design for fixed protein backbones |
| [NequIP](/papers/architectures/nequip) | molecular and materials potentials | E(3)-equivariant interatomic potential for energy-force modeling |
| [MEET](/papers/protein-modeling/meet-equivariant-peptide) | protein and geometric modeling | equivariant peptide modeling |
| [Multi-scale ML for Antibody-Antigen Binding](/papers/protein-modeling/multi-scale-antibody-binding) | protein interaction modeling | antibody-antigen binding evaluation |
| [Molexar](/papers/generative-models/molexar) | molecular generation | molecule-focused generation and representation |

## Concepts

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/data-evaluation|Data and Evaluation]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/protein-modeling/index|Protein modeling]]
- [[concepts/molecular-modeling/index|Molecular modeling]]
- [[concepts/evaluation/leakage|Leakage]]

## Related

- [[papers/index|Papers]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/analysis/limitation-taxonomy|Limitation taxonomy]]
