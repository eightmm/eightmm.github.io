---
title: Protein Language Model
tags:
  - protein-modeling
  - sequence
  - language-model
---

# Protein Language Model

A protein language model treats an amino-acid sequence as a token sequence and learns statistical structure from large protein databases. It is useful for representation learning, variant effect prediction, function prediction, structure-aware transfer, and protein design.

For a protein sequence:

$$
s=(a_1,\ldots,a_L),
\qquad
a_i\in\mathcal{V}_{\mathrm{AA}}
$$

a language model learns either an autoregressive distribution:

$$
p_\theta(s)
=
\prod_{i=1}^{L}
p_\theta(a_i\mid a_{<i})
$$

or a masked-token objective:

$$
\mathcal{L}_{\mathrm{MLM}}
=
-
\sum_{i\in M}
\log p_\theta(a_i\mid s_{\setminus M})
$$

where $M$ is the set of masked residue positions.

## What It Learns

| Signal | Where it comes from | What it may support |
|---|---|---|
| residue statistics | large sequence databases | family, motif, and conservation features |
| contextual dependencies | neighboring and distant residues | function and variant effect prediction |
| evolutionary constraints | homologous sequence distribution | structure and contact-related transfer |
| token embeddings | hidden states | residue, domain, or chain representation |

A protein language model does not directly observe experimental binding, activity, or structure unless those are added through supervision, retrieval, templates, or downstream data.

## Representation Levels

Token embeddings:

$$
h_i = f_\theta(s)_i
$$

Sequence-level embedding:

$$
z_s
=
\operatorname{pool}(h_1,\ldots,h_L)
$$

Pair or contact features may be derived from attention maps, outer products, or additional heads, but those features require separate validation before being treated as structural evidence.

## Common Uses

| Use | Output | Main risk |
|---|---|---|
| residue annotation | $p(y_i\mid s)$ | label imbalance and homolog leakage |
| protein classification | $p(y\mid s)$ | family memorization |
| variant scoring | $\Delta \log p_\theta$ or supervised score | assay and species shift |
| structure prediction input | sequence/MSA/template features | template and database leakage |
| generative design | sampled sequences | validity is not functional success |

## Likelihood and Biological Fitness

Sequence likelihood is not the same as biological fitness:

$$
\log p_\theta(s)
\neq
\operatorname{fitness}(s)
$$

High likelihood can mean that a sequence is common under the training distribution, not that it binds, folds, expresses, or performs the desired function. A paper that uses likelihood as a design score should connect it to downstream evidence.

## Evaluation Checks

- Are train and test proteins separated by [[concepts/evaluation/protein-family-split|Protein family split]] or sequence identity?
- Is the model evaluated on sequence likelihood, representation transfer, structure, function, or design?
- Are database release dates controlled when the benchmark contains recent proteins?
- Does the model use only sequence, or also [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]], templates, predicted structures, or labels?
- Are metrics reported at the residue, sequence, family, or assay-row level?
- Does the claim rely on likelihood, supervised downstream performance, or experimental validation?

## Related

- [[entities/protein|Protein]]
- [[entities/sequence|Sequence]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
