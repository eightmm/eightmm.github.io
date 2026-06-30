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

## Query Semantics

A query is not only a drawing of atoms. It is a set of constraints:

$$
Q = (V_Q,E_Q,C_V,C_E)
$$

where $C_V$ and $C_E$ constrain atom and bond properties.

| Constraint | Example issue |
| --- | --- |
| atom type | element, wildcard, atom list, isotope |
| valence/degree | aromatic and charged atoms may differ after sanitization |
| aromaticity | toolkit aromaticity model changes matches |
| charge | protonation policy can create or remove hits |
| stereochemistry | unspecified stereo should not be treated as confirmed stereo |
| recursive SMARTS | query may encode hidden expert assumptions |

For public notes, store the intention of the query in prose, not only the SMARTS string.

## Search Boundary

| Operation | Output | Do Not Confuse With |
| --- | --- | --- |
| Substructure search | boolean match or atom mappings | molecular similarity |
| SMARTS alert | query-defined structural pattern | toxicity mechanism proof |
| Scaffold filter | retained core or matched series | scaffold split unless policy is explicit |
| Fragment match | occurrence of a fragment | functional equivalence |
| Similarity search | ranked approximate neighbors | exact subgraph containment |

## Dataset Use

Substructure search can be a data operation, a scientific hypothesis, or a leakage source.

| Use | Good for | Caveat |
| --- | --- | --- |
| alert filtering | removing known reactive or undesirable motifs | alert hit is not proof of toxicity |
| scaffold grouping | series analysis or split construction | query policy must be fixed before evaluation |
| motif enrichment | explaining model errors or clusters | post-hoc query mining can overfit test labels |
| candidate retrieval | finding molecules with a required group | exact match may miss bioisosteres |
| data audit | locating duplicated cores or shortcuts | audit should not become hidden training signal |

## Practical Checks

- Is the query a strict substructure, scaffold, pharmacophore-like pattern, or loose SMARTS pattern?
- Were molecules standardized before searching?
- Are tautomers, protonation states, salts, and stereochemistry handled consistently?
- Are matches used for filtering, labeling, splitting, or interpretation?
- Could the query encode a known leakage shortcut?
- Are atom mappings, match counts, or only boolean hits used downstream?
- Was the query authored before looking at test labels or benchmark failures?
- Is the toolkit aromaticity and SMARTS behavior documented?
- Are query hits treated as hypotheses rather than biological proof?

## Related

- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
