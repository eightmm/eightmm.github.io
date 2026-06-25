---
title: Protein-Ligand Split
tags:
  - sbdd
  - evaluation
  - leakage
---

# Protein-Ligand Split

A protein-ligand split defines how complexes, proteins, ligands, assays, and structures are separated across train, validation, and test sets. It is the core benchmark contract for structure-based AI.

For a complex $c_i=(P_i,L_i,A_i,X_i)$, where $P_i$ is a protein or pocket, $L_i$ is a ligand, $A_i$ is assay or label context, and $X_i$ is structure or pose information, a split function is:

$$
s(c_i)\in\{\mathrm{train},\mathrm{val},\mathrm{test}\}
$$

The split should match the generalization claim.

## Split Axes

| Split axis | Held-out unit | Claim tested | Common leakage risk |
| --- | --- | --- | --- |
| Ligand-side | scaffold, analog cluster, similarity group | new chemotypes for known targets | close analogs across split |
| Protein-side | sequence family, structure family, target class | new targets or homolog-shift | homologs across split |
| Complex-side | protein-ligand pair or binding mode | new pairings | same pocket and close ligand seen separately |
| Assay/source | assay, campaign, lab, dataset source | cross-assay or cross-source robustness | protocol identity encoded in labels |
| Temporal | publication or measurement time | future-data generalization | retrospective curation choices |

## Grouped Split Constraint

Ligand grouping requires:

$$
g_L(L_i)=g_L(L_j)
\Rightarrow
s(c_i)=s(c_j)
$$

Protein grouping requires:

$$
g_P(P_i)=g_P(P_j)
\Rightarrow
s(c_i)=s(c_j)
$$

where $g_L$ may be scaffold or molecular similarity grouping, and $g_P$ may be sequence identity, protein family, structure family, or target class.

For strict protein-ligand generalization, both constraints may be required. For a narrow interpolation benchmark, the split can be looser, but the claim must also be narrower.

## Benchmark Questions

- Pose prediction: is the held-out complex separated from close ligand analogs, homologous targets, and template structures?
- Affinity prediction: are assay context, target family, ligand scaffold, and label semantics separated enough for the claim?
- Virtual screening: are decoys realistic and are ligand-only shortcuts controlled?
- Generative design: are generated molecules evaluated against held-out targets without reusing target-derived pose information?
- Interaction modeling: are binding sites, ligand scaffolds, and interaction fingerprints leaking across split?

## Checks

- What is the example unit: molecule, protein, complex, pose, assay row, or target-ligand pair?
- What is the split unit for proteins and for ligands?
- Are standardized molecules used before scaffold or similarity grouping?
- Are protein chains, isoforms, fragments, and homologs grouped before splitting?
- Are bound complexes, template databases, close analogs, or known poses excluded from test-time leakage?
- Are pose quality, affinity, enrichment, and generation claims evaluated under compatible split policies?
- Is the split contract reproducible from public fields rather than private curation notes?

## Related

- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
