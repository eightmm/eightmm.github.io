---
title: Protein Structure Cleaning
tags:
  - protein-modeling
  - structure
  - data-preprocessing
---

# Protein Structure Cleaning

Protein structure cleaning converts raw PDB/mmCIF records into model-ready structure inputs. It should be treated as part of the data protocol, not as invisible preprocessing.

A cleaning pipeline can be written as:

$$
S_{\mathrm{raw}}
\xrightarrow{\phi_{\mathrm{clean}}}
S_{\mathrm{model}}
$$

where $\phi_{\mathrm{clean}}$ includes chain selection, residue mapping, alternate-location handling, missing-residue masks, heteroatom policy, and coordinate normalization.

## Core Decisions

- Chain selection: which biological chain or assembly is used?
- AltLoc handling: which alternate conformation is kept?
- Missing residues: are gaps masked, modeled, or removed?
- Non-standard residues: are they mapped to canonical residues or marked unknown?
- Heteroatoms: are waters, ions, cofactors, glycans, and ligands retained?
- Hydrogen atoms: are hydrogens added, stripped, or ignored?
- Coordinate units and frame: are coordinates in Å and in the intended frame?

## Structure Identity

A cleaned structure should keep a stable identity key:

$$
I(S)
=
H(
\text{structure source},
\text{chain policy},
\text{residue policy},
\text{heteroatom policy},
\text{coordinate version}
)
$$

This matters for feature caching, split checks, and reproducibility.

## Interaction and Pocket Tasks

For structure-based tasks, cleaning choices can change the binding site:

- stripping a cofactor can remove a real interaction partner
- keeping a crystallographic ligand can leak the binding site
- removing waters can remove water-mediated contacts
- changing protonation can change hydrogen-bond patterns
- selecting a ligand-defined pocket can leak test information if not available at deployment

These choices should be documented in public notes at the protocol level without exposing private target details.

## Checks

- Is chain selection explicit?
- Are missing residues and chain breaks represented?
- Are waters, ions, cofactors, and ligands handled with a stated policy?
- Are non-standard residues mapped consistently?
- Is residue indexing preserved after cleaning?
- Is pocket definition independent of unavailable test-time information?
- Is the cleaned structure hash tied to downstream features?

## Related

- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
