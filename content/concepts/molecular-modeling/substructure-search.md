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

Atom constraints must also be preserved:

$$
c_Q(i) \preceq c_m(\phi(i))
$$

where $c_Q(i)$ is the query constraint for query atom $i$ and $c_m$ is the molecule atom state. The relation $\preceq$ depends on the query language, such as SMARTS.

## Key Ideas

- Substructure search is exact pattern matching under a chemical query language.
- It is different from similarity search, which ranks approximate neighbors.
- Query semantics matter: aromaticity, valence, charge, stereochemistry, and atom lists can change results.
- Substructure filters are useful for motifs, alerts, scaffolds, and dataset audits.

## Search Boundary

| Operation | Output | Do Not Confuse With |
| --- | --- | --- |
| Substructure search | boolean match or atom mappings | molecular similarity |
| SMARTS alert | query-defined structural pattern | toxicity mechanism proof |
| Scaffold filter | retained core or matched series | scaffold split unless policy is explicit |
| Fragment match | occurrence of a fragment | functional equivalence |
| Similarity search | ranked approximate neighbors | exact subgraph containment |

## Practical Checks

- Is the query a strict substructure, scaffold, pharmacophore-like pattern, or loose SMARTS pattern?
- Were molecules standardized before searching?
- Are tautomers, protonation states, salts, and stereochemistry handled consistently?
- Are matches used for filtering, labeling, splitting, or interpretation?
- Could the query encode a known leakage shortcut?
- Are atom mappings, match counts, or only boolean hits used downstream?
- Was the query authored before looking at test labels or benchmark failures?

## Related

- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
