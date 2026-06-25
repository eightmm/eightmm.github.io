---
title: Substructure Search
tags:
  - molecular-modeling
  - cheminformatics
  - graph
---

# Substructure Search

Substructure search asks whether a molecule contains a query pattern. It is a graph matching problem over atoms, bonds, aromaticity, charge, stereochemistry, and query constraints.

At a high level, the question is:

$$
Q \subseteq G_m
$$

where $Q$ is a query graph and $G_m$ is the molecular graph.

More explicitly, search asks whether there is an injective mapping $\phi$ from query atoms to molecule atoms such that atom and bond constraints are preserved:

$$
(i,j)\in E_Q
\Rightarrow
(\phi(i),\phi(j))\in E_m
$$

## Key Ideas

- Substructure search is exact pattern matching under a chemical query language.
- It is different from similarity search, which ranks approximate neighbors.
- Query semantics matter: aromaticity, valence, charge, stereochemistry, and atom lists can change results.
- Substructure filters are useful for motifs, alerts, scaffolds, and dataset audits.

## Practical Checks

- Is the query a strict substructure, scaffold, pharmacophore-like pattern, or loose SMARTS pattern?
- Were molecules standardized before searching?
- Are tautomers, protonation states, salts, and stereochemistry handled consistently?
- Are matches used for filtering, labeling, splitting, or interpretation?
- Could the query encode a known leakage shortcut?

## Related

- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
