---
title: ProteinMPNN
aliases:
  - papers/proteinmpnn
  - papers/protein-mpnn
  - papers/robust-deep-learning-based-protein-sequence-design
tags:
  - papers
  - architectures
  - protein-modeling
  - computational-biology
---

# ProteinMPNN

> The paper introduced ProteinMPNN, a message-passing neural network for protein sequence design conditioned on a fixed backbone structure.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Robust deep learning-based protein sequence design using ProteinMPNN |
| Authors | Justas Dauparas, Ivan Anishchenko, Nathaniel Bennett, et al. |
| Year | 2022 |
| Venue | Science |
| DOI | [10.1126/science.add2187](https://doi.org/10.1126/science.add2187) |
| Preprint | [bioRxiv 2022.06.03.494563](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1) |
| Code | [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) |
| Status | full note started |

## One-Line Takeaway

ProteinMPNN turns inverse protein folding into conditional autoregressive sequence design over a structure graph: given backbone coordinates and residue constraints, it predicts amino acid identities with message passing over local 3D neighborhoods.

## Question

Protein structure prediction asks:

$$
\text{sequence}
\rightarrow
\text{structure}.
$$

Protein sequence design often needs the reverse direction:

$$
\text{target backbone structure}
\rightarrow
\text{compatible sequence}.
$$

The architecture question is:

$$
\text{How do we design a sequence for a given protein backbone without expensive physics-heavy search at every step?}
$$

ProteinMPNN answers with a graph neural network trained to model:

$$
p_\theta(a_1,\ldots,a_L \mid X, C),
$$

where $a_i$ is the amino acid at residue $i$, $X$ is backbone geometry, and $C$ denotes optional constraints such as fixed residues, tied residues, chains, or design masks.

## Main Claim

A structure-conditioned message-passing model can design protein sequences with strong sequence recovery and broad practical utility across monomers, oligomers, nanoparticles, and binders.

The durable architecture pattern is:

$$
\text{backbone coordinates}
\rightarrow
\text{residue graph}
\rightarrow
\text{message passing}
\rightarrow
\text{autoregressive amino-acid decoding}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | protein backbone coordinates and chain/design constraints |
| Node unit | residue |
| Edge unit | residue-residue geometric neighborhood |
| Output | amino acid distribution per designed position |
| Main operator | message passing neural network over protein backbone graph |
| Decoding | order-agnostic autoregressive sequence prediction |
| Task | inverse folding / fixed-backbone protein sequence design |
| Domain | protein design and structure-conditioned sequence modeling |

The model is not primarily a structure generator. It assumes a backbone is given and designs sequences compatible with it:

$$
X
\text{ fixed}
\quad\Rightarrow\quad
a \sim p_\theta(a\mid X).
$$

## Graph Representation

ProteinMPNN represents a backbone as a graph:

$$
G=(V,E),
$$

where each node $i\in V$ is a residue and each edge $(i,j)\in E$ connects residues that are near each other in 3D or relevant by sequence/chain context.

The model consumes geometric features derived from backbone coordinates:

$$
X_i
=
\{N_i,C_{\alpha i},C_i,O_i\}
$$

for full-backbone models, with variants that can use less detailed geometry.

A residue-level message passing layer can be read abstractly as:

$$
m_{ij}^{(t)}
=
\phi_m
\left(
h_i^{(t)},
h_j^{(t)},
e_{ij}
\right),
$$

$$
h_i^{(t+1)}
=
\phi_u
\left(
h_i^{(t)},
\sum_{j\in\mathcal{N}(i)}m_{ij}^{(t)}
\right).
$$

Here $h_i$ is the residue hidden state and $e_{ij}$ encodes geometric/chain/relative features for the residue pair.

## Conditional Sequence Model

The target distribution is over amino acid identities:

$$
p_\theta(a\mid X)
=
\prod_{i=1}^{L}
p_\theta(a_{\pi_i}\mid a_{\pi_{<i}},X),
$$

where $\pi$ is a decoding order.

ProteinMPNN uses order-agnostic decoding: during training, decoding orders are sampled rather than fixed left-to-right. This matters because protein design is not naturally a text sequence problem.

For language modeling, left-to-right order is natural:

$$
p(x_1,\ldots,x_L)
=
\prod_i p(x_i\mid x_{<i}).
$$

For protein design on a fixed backbone, spatial contacts are not aligned with sequence order. Randomized or constraint-aware decoding lets the model condition on arbitrary known residues:

$$
p_\theta(a_U\mid a_K,X),
$$

where $K$ are fixed or already decoded positions and $U$ are positions to design.

## Why Message Passing Fits Protein Design

Residue identity depends on both local chemistry and nonlocal contacts:

| Signal | Example | Why graph modeling helps |
| --- | --- | --- |
| local backbone geometry | helix, strand, loop | constrains allowed residue preferences |
| packing environment | buried vs exposed | changes hydrophobic/polar preference |
| side-chain contact context | neighboring residues | sequence choices are coupled |
| oligomer/interface context | inter-chain contacts | binder and assembly design depend on cross-chain geometry |
| fixed residues | motif or functional site | model must condition around constraints |

Message passing lets information move across the structure graph:

$$
h_i^{(T)}
=
f_\theta
\left(
X_i,\{X_j:j\in\mathcal{N}^T(i)\}
\right).
$$

The model can therefore design residue $i$ using local structure and multi-hop contact context.

## Order-Agnostic Autoregression

ProteinMPNN is often remembered for being practical, not just for being a graph neural network.

The order-agnostic decoding setup supports:

| Design situation | Why fixed left-to-right decoding is awkward |
| --- | --- |
| fixed motif residues | known residues may occur in the middle of the chain |
| multi-chain design | chains do not form one natural sequence order |
| tied residues | symmetry or repeat constraints couple distant positions |
| partial redesign | some positions are fixed while others are sampled |

The conditional sequence objective can be read as:

$$
\mathcal{L}(\theta)
=
-\sum_{i\in U}
\log p_\theta(a_i\mid a_K,a_{\pi_{<i}},X).
$$

This objective is architecture-relevant because it shapes the information flow: the model must combine structural context with already-known sequence context.

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [GVP](/papers/architectures/geometric-vector-perceptrons) | protein structure graph learning | GVP emphasizes scalar/vector equivariant features; ProteinMPNN emphasizes inverse-folding sequence design |
| [AlphaFold2](/papers/architectures/alphafold2) | protein sequence/structure reasoning | AlphaFold2 predicts structure from sequence/MSA; ProteinMPNN designs sequence from structure |
| [AlphaFold3](/papers/architectures/alphafold3) | biomolecular structure modeling | AlphaFold3 predicts complexes; ProteinMPNN designs sequences for a given protein backbone |
| [MPNN](/papers/architectures/neural-message-passing-for-quantum-chemistry) | message passing over structured objects | ProteinMPNN applies residue-level message passing to protein inverse folding |
| [DimeNet](/papers/architectures/dimenet) | geometric message passing | DimeNet targets atomistic molecular properties with directional basis functions |
| [SE(3)-Transformer](/papers/architectures/se3-transformer) | 3D structure graph learning | SE(3)-Transformer guarantees equivariant attention; ProteinMPNN is a practical protein design model |

## Evidence to Read

| Evidence | What it supports | What it does not prove |
| --- | --- | --- |
| native-backbone sequence recovery | model captures strong structure-to-sequence signal | sequence recovery alone is not functional validation |
| comparison to Rosetta sequence design | neural design can outperform a physics-heavy baseline on reported recovery | every design task is solved by ProteinMPNN |
| experimental characterization | designed sequences can fold/function in several reported settings | arbitrary target or binder design will work without careful protocol |
| fixed/tied residue support | architecture is useful for constrained design workflows | constraints remove the need for downstream validation |

## Why This Matters

ProteinMPNN became a standard component in protein design pipelines because it fills a specific role:

$$
\text{given a backbone, sample plausible sequences quickly}.
$$

That role is different from:

| Model type | Direction |
| --- | --- |
| structure predictor | sequence $\rightarrow$ structure |
| backbone generator | noise or constraints $\rightarrow$ structure |
| sequence designer | structure $\rightarrow$ sequence |
| evaluator | structure/sequence $\rightarrow$ score |

In modern design workflows, a backbone generator or hand-designed scaffold may produce $X$, ProteinMPNN samples candidate sequences $a$, and a structure predictor/evaluator checks whether $a$ folds back to the intended structure.

## Limitations

ProteinMPNN is conditioned on a backbone. It does not by itself solve backbone generation, conformational ensemble modeling, ligand-aware design, or wet-lab validation.

High sequence recovery does not equal biological function. A sequence can match native-like preferences but still fail due to dynamics, expression, solubility, binding kinetics, off-target interactions, or assay context.

The model's behavior depends on the quality and distribution of backbone structures used for training and inference. If the input backbone is unrealistic, overconstrained, or missing important state information, the sequence samples can still be misleading.

## Common Misreadings

### "ProteinMPNN is a protein language model."

No. It is structure-conditioned. A protein language model primarily models sequences; ProteinMPNN models sequence choices conditioned on backbone geometry.

### "ProteinMPNN designs proteins end to end."

No. It designs sequences for a provided backbone. End-to-end design usually includes backbone generation, sequence design, structure prediction, filtering, and experimental validation.

### "Sequence recovery is the same as design success."

No. Sequence recovery measures agreement with native residues under a benchmark setup. Design success requires the sequence to fold and function under the intended conditions.

## What to Remember

When reading protein design papers that use ProteinMPNN, ask:

- Is ProteinMPNN the main method or only a sequence-design module?
- What backbone generated the design target?
- Which residues were fixed, tied, biased, or masked?
- Was the design evaluated with an independent structure predictor?
- Is the claim about sequence recovery, folding, binding, function, or experimental validation?
- Are failures hidden by filtering many sampled sequences?

The compact mental model is:

$$
\text{ProteinMPNN}
=
\text{residue graph message passing}
+
\text{structure-conditioned autoregressive sequence design}.
$$

## Links

- [[papers/architectures/geometric-vector-perceptrons|Geometric Vector Perceptrons]]
- [[papers/architectures/neural-message-passing-for-quantum-chemistry|Neural Message Passing]]
- [[papers/architectures/alphafold2|AlphaFold2]]
- [[papers/architectures/alphafold3|AlphaFold3]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/index|Protein modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
