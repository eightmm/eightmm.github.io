---
title: Protein Modeling
aliases:
  - computational-biology/protein-modeling
  - research/protein-modeling
tags:
  - computational-biology
  - protein-modeling
unlisted: true
---

# Protein Modeling

Protein modeling notes cover folding, structure prediction, protein representation learning, and links to downstream structure-based tasks.

A broad protein modeling view is:

$$
\hat{z},\hat{X},\hat{y} = f_\theta(s, X_{\mathrm{obs}}, c)
$$

where $s$ is a protein sequence, $X_{\mathrm{obs}}$ is optional observed or predicted structure, $c$ is optional context, and the output can be a representation $\hat{z}$, coordinates $\hat{X}$, or a task label $\hat{y}$.

## Route Map

| Question | Start |
| --- | --- |
| What object is being modeled? | [Protein](/entities/protein), [Sequence](/entities/sequence), [Structure](/entities/structure) |
| How is the protein represented? | [Protein representation](/concepts/protein-modeling/protein-representation), [Protein language model](/concepts/protein-modeling/protein-language-model), [Sequence](/concepts/modalities/sequence), [3D structure](/concepts/modalities/3d-structure) |
| Does the method use evolutionary context? | [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment), [Protein domain](/concepts/protein-modeling/protein-domain) |
| Does the method predict coordinates or contacts? | [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map), [Coordinate prediction](/concepts/tasks/coordinate-prediction) |
| Is residue numbering or chain mapping important? | [Residue indexing](/concepts/protein-modeling/residue-indexing), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) |
| Is this linked to binding or docking? | [Structure-based modeling](/molecular-modeling/structure-based), [Protein-ligand complex](/entities/protein-ligand-complex), [Protein-ligand interaction](/concepts/sbdd/protein-ligand-interaction) |
| What split tests transfer? | [Protein family split](/concepts/evaluation/protein-family-split), [OOD generalization](/concepts/evaluation/ood-generalization) |

## Representation Levels

| Level | Unit | Common Representation | Common Claim |
| --- | --- | --- | --- |
| token | amino acid residue | token embedding, logits | masked residue prediction, residue annotation |
| segment | motif, domain, region | pooled embedding, local features | function, site, domain classification |
| chain | full protein sequence or structure | sequence embedding, structure embedding | family, function, stability, localization |
| pocket | binding site residues and coordinates | pocket graph, grid, local frame | ligand binding, docking context |
| complex | protein plus partner | interaction graph, pair representation | binding, pose, interface, affinity |

The same architecture can mean different things at different levels. A Transformer over residues is not the same claim as a graph model over a protein-ligand complex.

## Modeling Modes

| Mode | Typical Input | Typical Output | Evaluation Risk |
| --- | --- | --- | --- |
| sequence representation | sequence, MSA, domain annotation | embedding, residue features | homolog leakage, family imbalance |
| structure prediction | sequence, template, MSA | coordinates, distances, contacts | template leakage, atom/residue mapping |
| functional prediction | sequence or structure | class, site, activity, property | label semantics, assay/source leakage |
| interaction modeling | protein plus ligand, peptide, protein, or nucleic acid | binding score, pose, contact, interface | split must control both interaction sides |
| generative design | sequence, backbone, pocket, constraints | sequence, structure, complex, design candidates | validity, novelty, diversity, downstream assay gap |

## Sequence, Structure, and Context

Protein papers often mix three different information sources:

$$
s = (a_1,\ldots,a_L),
\qquad
X = (x_1,\ldots,x_L),
\qquad
c = \{\text{MSA},\text{template},\text{pocket},\text{assay},\text{species}\}
$$

Keep these separate when reading a method:

| Source | What It Provides | Risk |
| --- | --- | --- |
| sequence | residue order and identity | homolog leakage and duplicate sequences |
| MSA | evolutionary co-variation | train/test family overlap and database time leakage |
| template | structural prior | template leakage into structure benchmarks |
| predicted structure | coordinates with model bias | circular evidence if used as ground truth-like input |
| pocket or ligand context | local interaction condition | ligand-defined pocket leakage |
| assay or function label | measured endpoint | source, organism, construct, and label ambiguity |

## Common Objectives

| Objective | Sketch | What It Trains |
| --- | --- | --- |
| masked residue prediction | $-\log p_\theta(a_i\mid s_{\setminus i})$ | sequence representation |
| autoregressive protein modeling | $-\sum_i \log p_\theta(a_i\mid a_{<i})$ | sequence distribution and generation |
| contrastive protein learning | $-\log \frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}{\sum_j \exp(\operatorname{sim}(z_i,z_j)/\tau)}$ | representation invariance |
| distance/contact prediction | $\ell(\hat{d}_{ij}, d_{ij})$ or BCE over contacts | pair geometry |
| coordinate prediction | $\lVert \hat{X}-X\rVert$ after an alignment policy | structure geometry |
| functional prediction | $\ell(f_\theta(s,X), y)$ | downstream task behavior |

The objective should be linked to the evidence. A good language-modeling loss does not automatically prove structure, binding, or functional transfer.

For sequence-only models, the distinction is:

$$
\text{sequence likelihood}
\rightarrow
\text{representation or generation claim}
\not\Rightarrow
\text{biological function claim}
$$

Function, binding, stability, and design claims need downstream labels, structural evidence, or experimental validation.

## Evidence Fields

| Field | Why It Matters |
| --- | --- |
| sequence source | duplicate proteins and homologs define leakage risk |
| structure source | experimental, predicted, template-derived, and relaxed structures support different claims |
| representation unit | token, residue, domain, chain, complex, pocket, or full structure changes the task |
| split unit | row split, protein family split, fold split, target split, and complex split test different generalization claims |
| metric | sequence, coordinate, ranking, binding, and generation metrics are not interchangeable |
| uncertainty | replicate noise, structural ambiguity, seed variance, and family imbalance affect interpretation |

## Paper Note Checklist

| Check | Question |
| --- | --- |
| object | Is the unit residue, chain, pocket, complex, family, or assay row? |
| representation | Is the model using sequence, MSA, predicted structure, experimental structure, or ligand context? |
| split | Does the split remove homologs, related structures, and shared interaction partners? |
| template policy | Are templates, predicted structures, and database dates controlled? |
| metric | Does the metric match the claimed output: sequence, contact, coordinate, binding, or function? |
| downstream gap | Does the benchmark actually test the biological or chemical use case? |

## Adjacent Areas

- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/protein-modeling/protein-language-model|Protein language model]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[ai/generative-models|Generative models]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/protein-modeling/index|Protein modeling papers]]
