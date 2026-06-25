---
title: Stereochemistry
tags:
  - molecular-modeling
  - chemistry
---

# Stereochemistry

Stereochemistry describes the 3D arrangement of atoms that can make molecules with the same graph behave differently. It matters for binding, activity, conformers, and molecular generation.

A modeling pipeline should treat stereochemistry as part of the molecular identity:

$$
M_{\mathrm{id}}
=
\operatorname{standardize}
\left(
\operatorname{topology},
\operatorname{stereo},
\operatorname{charge},
\operatorname{tautomer}
\right)
$$

## Checks

- Are chiral centers and double-bond stereochemistry preserved?
- Are unspecified stereocenters treated as unknown rather than silently flattened?
- Does featurization distinguish stereoisomers when the task requires it?
- Are generated molecules checked for valid stereochemistry?
- Is stereochemistry standardized before deduplication and split assignment?

## Related

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[entities/molecule|Molecule]]
