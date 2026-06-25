---
title: MambaFold
tags:
  - protein-folding
  - sequence-modeling
  - mamba
---

# MambaFold

MambaFold is a research direction for protein structure modeling with state-space sequence models. The main question is how much long-range structural signal can be captured without relying only on attention-heavy architectures.

One abstract pipeline is:

$$
h_{1:L} = \operatorname{Mamba}_\theta(s_{1:L}),
\qquad
\hat{X} = \operatorname{Decoder}_\phi(h_{1:L})
$$

Here $s_{1:L}$ is a residue sequence, $h_{1:L}$ are sequence states, and $\hat{X}$ is a structural prediction or geometry-aware representation.

## Research Angles

- Sequence representation for residues, domains, and evolutionary context.
- Structural constraints for backbone geometry.
- Interfaces with [[concepts/geometric-deep-learning/equivariant-gnn|equivariant GNN]] layers or geometry-aware decoders.
- Connections to [[research/self-supervised-learning/index|self-supervised learning]] for protein representations.
- Evaluation against folding, docking, and downstream protein-ligand tasks.

## Open Notes

- Separate architecture notes from benchmark results.
- Keep unpublished experiments out of public pages.
- Link stable concepts into [[research/llm-wiki|LLM Wiki]] pages as they mature.

## Related

- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[research/protein-modeling/index|Protein modeling]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
