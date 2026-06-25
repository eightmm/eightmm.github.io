---
title: Sequence
tags:
  - modalities
  - sequence
---

# Sequence

A sequence is an ordered list of tokens, symbols, events, frames, residues, or measurements. Text, protein sequences, genome segments, time series, audio frames, and action traces can all be represented as sequences, but they do not share the same semantics.

A generic sequence is:

$$
x_{1:T} = (x_1, x_2, \ldots, x_T)
$$

where $T$ is sequence length and $x_t$ is the item at position $t$.

## Conditional Modeling

Autoregressive sequence modeling factorizes a joint distribution into conditionals:

$$
p(x_{1:T})
=
\prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

Masked or denoising sequence modeling predicts hidden positions from context:

$$
\mathcal{L}
=
-\sum_{t \in \mathcal{M}}
\log p_\theta(x_t \mid x_{\setminus \mathcal{M}})
$$

where $\mathcal{M}$ is the masked index set.

## Important Axes

- Alphabet: byte, subword, amino acid, nucleotide, event type, or continuous token.
- Length: short fixed sequence, long document, long genome region, or streaming signal.
- Directionality: causal, bidirectional, local-window, or global context.
- Alignment: positions may correspond across modalities, homologs, frames, or timestamps.
- Structure: sequences may imply graph, geometry, hierarchy, or repeated motifs.

## Checks

- What does one token represent?
- Is position absolute, relative, periodic, temporal, or biologically indexed?
- Can the model see future positions?
- Is truncation removing the important signal?
- Are near-duplicate or homologous sequences split across train and test?

## Related

- [[concepts/modalities/text|Text]]
- [[entities/sequence|Sequence entity]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/learning/masked-modeling|Masked modeling]]
