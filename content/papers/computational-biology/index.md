---
title: Computational Biology Papers
tags:
  - papers
  - computational-biology
---

# Computational Biology Papers

Computational Biology paper notes cover structure-based modeling, protein modeling, molecular generation, protein-ligand interaction, assay-aware labels, and domain-specific evaluation.

This is the visible paper shelf for biology-facing papers. Narrower buckets such as [[papers/sbdd/index|Structure-based modeling papers]] and [[papers/protein-modeling/index|Protein modeling papers]] remain as support buckets so existing paper URLs stay stable.

## Reading Axes

- What object is modeled: protein, molecule, ligand, pocket, complex, pose, assay record, or sequence region?
- What route is central: [[molecular-modeling/sequence-based|sequence-based modeling]], [[molecular-modeling/molecular-ligand|molecular and ligand modeling]], [[molecular-modeling/interactions|interaction modeling]], or [[molecular-modeling/structure-based/index|structure-based modeling]]?
- Is the contribution an AI method, a domain workflow, a benchmark, a dataset, or an evaluation protocol?
- Are molecule standardization, protein-family split, scaffold split, assay semantics, and leakage risks explicit?
- Does the paper separate method quality from benchmark artifacts or domain shortcuts?

## Sub-Buckets

| Bucket | Use For | Notes |
| --- | --- | --- |
| [Structure-based modeling papers](/papers/sbdd) | docking, pose quality, scoring, conformers, virtual screening | [PoseBusters](/papers/sbdd/posebusters) |
| [Protein modeling papers](/papers/protein-modeling) | protein representation, structure, antibody/protein interaction | [MEET](/papers/protein-modeling/meet-equivariant-peptide), [Multi-scale antibody binding](/papers/protein-modeling/multi-scale-antibody-binding) |
| [Generative model papers](/papers/generative-models) | molecule/protein generation when generation is the strongest claim | [Molexar](/papers/generative-models/molexar) |

## Active Notes

| Paper Note | Main Route | Why It Belongs Here |
| --- | --- | --- |
| [PoseBusters](/papers/sbdd/posebusters) | structure-based evaluation | pose plausibility and docking evaluation |
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
