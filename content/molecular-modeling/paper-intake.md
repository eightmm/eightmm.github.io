---
title: Molecular Modeling Paper Intake
aliases:
  - bio/paper-intake
tags:
  - molecular-modeling
  - papers
---

# Molecular Modeling Paper Intake

Molecular modeling 논문은 model score보다 object, label, split, leakage를 먼저 고정해야 합니다. 특히 molecule, protein, ligand, pocket, conformer, complex를 다루는 논문은 row 하나가 무엇을 의미하는지부터 확인해야 합니다.

$$
\text{bio example}
=
(\text{object}, \text{context}, \text{label}, \text{split unit})
$$

## Intake Fields

| Field | Question | Route |
| --- | --- | --- |
| Object | Is the example a molecule, ligand, protein, pocket, complex, pose, assay record, or genome region? | [Entities](/molecular-modeling/entities) |
| Representation | Is it SMILES, graph, fingerprint, sequence, embedding, conformer, coordinate, or complex graph? | [Molecules](/molecular-modeling/molecules), [Proteins](/molecular-modeling/proteins), [Geometry](/molecular-modeling/geometry) |
| Chemical state | Are salt, stereochemistry, tautomer, protonation, charge, and conformer policy fixed? | [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Context | Does prediction depend on target, assay, pocket, receptor state, species, construct, or template? | [Computational Biology](/molecular-modeling/computational-biology) |
| Label | What endpoint, unit, direction, threshold, censoring, replicate rule, and source define the target? | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Split | What unit is held out: scaffold, sequence family, complex pair, assay/source, time, or template? | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Metric | Is the claim about affinity, pose, ranking, enrichment, validity, novelty, or property prediction? | [Evaluation](/ai/evaluation) |
| Benchmark claim | Does the benchmark define data, task, split, metric, allowed information, and reporting rule? | [Benchmark intake](/concepts/data/benchmark-intake) |
| Public boundary | Are data, metadata, artifacts, and claims public and non-sensitive? | [Papers](/papers) |

## Domain Risk Map

| Area | Common Failure | Check |
| --- | --- | --- |
| Molecules | salts, tautomers, protonation, stereo, and duplicates change identity | standardize before dedup and split |
| Proteins | homologs or near-identical sequences leak across train/test | sequence-identity or family split |
| Structures | templates, bound ligands, or close analogs make the test task easier | template and analog leakage check |
| Assays | pooled measurements mix incompatible endpoints or conditions | target-assay-label contract |
| Docking | pose success may depend on known pocket or ligand-defined frame | inference-time information boundary |
| Virtual screening | decoys may separate by trivial physicochemical properties | property-only baseline and enrichment metrics |

## Minimum Evidence

A molecular modeling paper note should record:

- Example unit: molecule, protein, complex, pose, assay record, or generated sample.
- Label semantics: endpoint, unit, direction, threshold, censoring, replicate aggregation, and source.
- Preprocessing: molecule standardization, protein cleaning, coordinate source, protonation, conformer rule.
- Chemical state: salt policy, stereo, tautomer, protonation, charge, conformer generation, and featurizer cache key.
- Split unit: scaffold, protein family, complex pair, assay/source, time, or template-aware split.
- Baseline: fingerprint/tree model, sequence similarity, docking score, or a task-specific simple model.
- Metric: primary decision metric plus calibration, uncertainty, enrichment, or failure diagnostics.
- Leakage check: duplicate, scaffold, homolog, template, assay/source, preprocessing, and coordinate-frame leakage.

## Structure-Based Shortcut

For structure-based modeling, ask this before trusting a score:

$$
\text{available at inference}
\subseteq
\text{available during evaluation}
$$

If evaluation uses a known ligand pose, ligand-defined pocket, homologous template, or close analog that would not be available in deployment, the result may not support the claimed generalization.

## Update Targets

- Molecule or ligand issue: [[molecular-modeling/molecules|Molecules]] and [[concepts/molecular-modeling/index|Molecular modeling]]
- Protein or sequence issue: [[molecular-modeling/proteins|Proteins]] and [[concepts/protein-modeling/index|Protein modeling]]
- Docking or SBDD issue: [[molecular-modeling/docking|Docking]] and [[concepts/sbdd/index|SBDD concepts]]
- Split or benchmark issue: [[molecular-modeling/data-evaluation|Data and Evaluation]]
- Benchmark contract issue: [[concepts/data/benchmark-intake|Benchmark intake]]
- General AI method issue: [[ai/paper-intake|AI paper intake]]
- Formula issue: [[math/formula-intake|Formula intake]]
- Multi-axis issue: [[papers/workflows/claim-routing|Claim routing]]

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
