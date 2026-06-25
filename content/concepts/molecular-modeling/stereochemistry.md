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
- Does a stereoisomer pair produce distinct features when labels can differ?

## Common Pitfall

If stereochemistry is dropped, two molecules with the same topology can collapse to one representation:

$$
F(M_R)=F(M_S)
$$

This may be acceptable only if the task explicitly ignores stereochemistry. For binding, docking, conformers, and many activity labels, this collapse can destroy the signal.

## Related

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[entities/molecule|Molecule]]
