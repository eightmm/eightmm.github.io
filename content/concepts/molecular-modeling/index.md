---
title: Molecular Modeling Concepts
tags:
  - molecular-modeling
  - concepts
---

# Molecular Modeling Concepts

Molecular modeling concepts describe how small molecules become model inputs: strings, graphs, fingerprints, conformers, descriptors, 3D coordinates, and physics-based geometry protocols.

The modeling object is not just a drawing of atoms. It is a stateful record:

$$
M =
(\text{topology}, \text{stereo}, \text{tautomer}, \text{protonation}, \text{conformer}, \text{source})
$$

Different choices can produce different deduplication keys, split assignments, features, poses, and labels.

## Workflow

Use this order for public ML notes:

$$
\text{raw record}
\rightarrow
\text{standardize}
\rightarrow
\text{define identity}
\rightarrow
\text{deduplicate}
\rightarrow
\text{split}
\rightarrow
\text{featurize}
\rightarrow
\text{model}
$$

Do not split or aggregate labels before the molecular identity policy is explicit.

## Route Map

| Question | Start | Risk |
| --- | --- | --- |
| What is the molecule identity? | [Molecular identity](/concepts/molecular-modeling/molecular-identity), [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) | salts, tautomers, stereo, protonation, charge, source policy |
| What does the model see? | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) | featurizer silently changing the object |
| How is similarity or retrieval defined? | [Molecular similarity](/concepts/molecular-modeling/molecular-similarity), [Substructure search](/concepts/molecular-modeling/substructure-search) | proxy similarity used as domain truth |
| What prediction task is claimed? | [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction) | label context and split unit missing |
| Is generation constrained? | [Fragment-SELFIES](/concepts/molecular-modeling/fragment-selfies) | valid strings without useful molecules |
| Is 3D state involved? | [Conformer](/concepts/molecular-modeling/conformer), [Force field](/concepts/molecular-modeling/force-field), [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | conformer source and postprocessing dependence |
| Which chemical variants matter? | [Tautomer](/concepts/molecular-modeling/tautomer), [Protonation state](/concepts/molecular-modeling/protonation-state), [Stereochemistry](/concepts/molecular-modeling/stereochemistry) | train/test leakage through equivalent or near-equivalent raw rows |

## Geometry and Physics Protocols

| Concept | Use For | Main Risk |
| --- | --- | --- |
| [Conformer](/concepts/molecular-modeling/conformer) | ligand 3D geometry and conformer ensembles | training and inference may use different conformer sources |
| [Force field](/concepts/molecular-modeling/force-field) | geometry energy, minimization, MD, clash checks | energy is model-dependent, not an absolute truth |
| [Energy minimization](/concepts/molecular-modeling/energy-minimization) | relaxing conformers or poses | postprocessing can hide invalid model outputs |
| [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | trajectories and flexible structure analysis | frame leakage and protocol dependence can dominate |

## Data Checks

- Standardize molecules before deduplication and splitting.
- Define molecular identity before deduplication, label aggregation, and split assignment.
- Decide whether to preserve or flatten stereochemistry.
- Record tautomer, salt, charge, protonation, and conformer protocols.
- Record force-field, minimization, and molecular-dynamics protocols when coordinates are generated or refined.
- Use scaffold or cluster splits instead of random splits for ligand-side generalization.
- Cache features with featurizer version and input hash.
- Use one molecular featurization contract across train, evaluation, and inference.

## Failure Modes

- Salt, tautomer, charge, or stereo variants leak across train/test as different raw rows.
- A graph featurizer drops chiral tags or bond stereo while the label distinguishes stereoisomers.
- 3D models train on crystal/bound conformations but deploy on generated conformers without measuring shift.
- Postprocessing with energy minimization changes model outputs but is reported as if it were the raw model.
- Similarity or scaffold splits are computed on raw molecules instead of standardized identity.
- A random split is used for a congeneric series, causing memorization to look like generalization.

## Related

- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[entities/target-assay-label|Target-assay-label]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
