---
title: Residue Indexing
tags:
  - protein-modeling
  - data
  - structure
---

# Residue Indexing

Residue indexing defines how sequence positions, structure residues, chain IDs, insertion codes, and model tokens refer to the same biological residue. It is a frequent source of silent bugs in protein modeling.

A protein sequence uses simple positions:

$$
i \in \{1,\ldots,L\}
$$

A structure file may identify a residue by:

$$
r = (\text{chain}, \text{resSeq}, \text{iCode}, \text{altLoc})
$$

These are not the same coordinate system. A safe pipeline defines an explicit map:

$$
\rho:
i_{\mathrm{seq}}
\rightarrow
r_{\mathrm{struct}}
\cup
\{\varnothing\}
$$

where $\varnothing$ means the sequence residue has no resolved structure residue.

## Why It Matters

Protein language model embeddings are indexed by sequence tokens. Structure coordinates are indexed by observed residues and atoms. PDB/mmCIF files can contain missing residues, insertion codes, non-standard residues, alternate locations, chain breaks, engineered tags, and numbering that does not start at 1.

If the pipeline matches the $i$-th sequence token to the $i$-th structure row, embeddings can be paired with the wrong coordinates without crashing.

## Common Cases

- Missing residues: sequence position exists, but no coordinate is present.
- Insertion code: residue numbers such as `82A` and `82B`.
- Alternate location: multiple conformations for the same atom or residue.
- Non-standard residue: modified residue mapped to a canonical or unknown token.
- Chain mismatch: structure contains several chains, but the model expects one chain.
- Special tokens: protein language models may add BOS/EOS/CLS tokens that should not be aligned to residues.

## Alignment Contract

A residue alignment record should include:

- source sequence ID
- structure ID or file ID
- chain ID
- sequence position
- structure residue identifier
- residue name in sequence
- residue name in structure
- missing or mapped status
- non-standard residue policy

The aligned representation can be written as:

$$
z_i
=
\left(
h_i,
x_{\rho(i)},
m_i
\right)
$$

where $h_i$ is a sequence embedding, $x_{\rho(i)}$ is a coordinate if present, and $m_i$ is a mask indicating whether structure information exists.

## Checks

- Is FASTA-to-structure alignment computed explicitly?
- Are insertion codes and chain IDs preserved in the residue key?
- Are missing residues represented with masks rather than shifted indices?
- Are special tokens excluded from residue-level embeddings?
- Are non-standard residues mapped or dropped with a stated rule?
- Does a sample alignment around missing-density regions look correct?

## Related

- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
