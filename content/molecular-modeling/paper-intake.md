---
title: Computational Biology Paper Intake
aliases:
  - computational-biology/paper-intake
  - bio/paper-intake
tags:
  - computational-biology
  - papers
unlisted: true
---

# Computational Biology Paper Intake

Computational biology 논문은 model score보다 object, label, split, leakage를 먼저 고정해야 합니다. 특히 molecule, protein, ligand, pocket, conformer, complex, genome region을 다루는 논문은 row 하나가 무엇을 의미하는지부터 확인해야 합니다.

$$
\text{bio example}
=
(\text{object}, \text{context}, \text{label}, \text{split unit})
$$

## Intake Fields

| Field | Question | Route |
| --- | --- | --- |
| Object | Is the example a molecule, ligand, protein, pocket, complex, pose, assay record, or genome region? | [Objects and Entities](/molecular-modeling/entities) |
| Representation | Is it SMILES, graph, fingerprint, sequence, embedding, conformer, coordinate, or complex graph? | [Molecular and Ligand Modeling](/molecular-modeling/molecular-ligand), [Sequence-Based Modeling](/molecular-modeling/sequence-based), [Geometry for Structure Modeling](/molecular-modeling/geometry-for-structure-modeling) |
| Protein sequence model | Is it a protein language model, MSA-based model, template-based model, or hybrid? | [Protein language model](/concepts/protein-modeling/protein-language-model), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) |
| Chemical state | Are salt, stereochemistry, tautomer, protonation, charge, and conformer policy fixed? | [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) |
| Context | Does prediction depend on target, assay, pocket, receptor state, species, construct, or template? | [Computational Biology Boundary](/molecular-modeling/computational-biology) |
| Label | What endpoint, unit, direction, threshold, censoring, replicate rule, and source define the target? | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Split | What unit is held out: scaffold, sequence family, complex pair, assay/source, time, or template? | [Data and Evaluation](/molecular-modeling/data-evaluation) |
| Metric | Is the claim about affinity, pose, ranking, probability, enrichment, validity, novelty, or property prediction? | [Evaluation](/ai/evaluation), [Probability metrics](/concepts/evaluation/probability-metrics) |
| Energy or constraint | Does the method use a force field, learned energy, score, minimization, projection, or validity filter? | [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Energy-based model](/concepts/generative-models/energy-based-model), [Constrained optimization](/concepts/math/constrained-optimization) |
| Claim pattern | Is it property prediction, activity prediction, protein representation, docking, generation, protein design, or genome sequence modeling? | [Computational Biology paper claim patterns](/molecular-modeling/paper-claim-patterns) |
| Benchmark claim | Does the benchmark define data, task, split, metric, allowed information, and reporting rule? | [Benchmark intake](/concepts/data/benchmark-intake) |
| Public boundary | Are data, metadata, artifacts, and claims public and non-sensitive? | [Papers](/papers) |

## Domain Risk Map

| Area | Common Failure | Check |
| --- | --- | --- |
| Molecules | salts, tautomers, protonation, stereo, and duplicates change identity | standardize before dedup and split |
| Proteins | homologs or near-identical sequences leak across train/test | sequence-identity or family split |
| Protein language models | sequence likelihood is mistaken for function, binding, or design success | downstream evidence and family split |
| Structures | templates, bound ligands, or close analogs make the test task easier | template and analog leakage check |
| Assays | pooled measurements mix incompatible endpoints or conditions | target-assay-label contract |
| Docking | pose success may depend on known pocket or ligand-defined frame | inference-time information boundary |
| Virtual screening | decoys may separate by trivial physicochemical properties | property-only baseline and enrichment metrics |

## Minimum Evidence

A computational biology paper note should record:

- Example unit: molecule, protein, complex, pose, assay record, or generated sample.
- Label semantics: endpoint, unit, direction, threshold, censoring, replicate aggregation, and source.
- Preprocessing: molecule standardization, protein cleaning, coordinate source, protonation, conformer rule.
- Chemical state: salt policy, stereo, tautomer, protonation, charge, conformer generation, and featurizer cache key.
- Sequence model source: PLM, MSA, template, predicted structure, or hybrid context when relevant.
- Split unit: scaffold, protein family, complex pair, assay/source, time, or template-aware split.
- Baseline: fingerprint/tree model, sequence similarity, docking score, or a task-specific simple model.
- Metric: primary decision metric plus calibration, uncertainty, enrichment, or failure diagnostics.
- Constraint accounting: hard constraints, penalties, projection, repair, filtering, and invalid-output denominator.
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

- Molecule or ligand issue: [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]] and [[concepts/molecular-modeling/index|Molecular modeling]]
- Protein or sequence issue: [[molecular-modeling/sequence-based|Sequence-based modeling]] and [[concepts/protein-modeling/index|Protein modeling]]
- Protein language model issue: [[concepts/protein-modeling/protein-language-model|Protein language model]]
- Docking or SBDD issue: [[molecular-modeling/docking|Docking]] and [[concepts/sbdd/index|SBDD concepts]]
- Energy, score, or constraint issue: [[concepts/generative-models/energy-based-model|Energy-based model]], [[concepts/molecular-modeling/energy-minimization|Energy minimization]], or [[concepts/math/constrained-optimization|Constrained optimization]]
- Split or benchmark issue: [[molecular-modeling/data-evaluation|Data and Evaluation]]
- Benchmark contract issue: [[concepts/data/benchmark-intake|Benchmark intake]]
- General AI method issue: [[ai/paper-intake|AI paper intake]]
- Formula issue: [[math/formula-intake|Formula intake]]
- Multi-axis issue: [[papers/workflows/claim-routing|Claim routing]]
- Repeated paper type: [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
