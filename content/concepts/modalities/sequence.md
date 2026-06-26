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

## Representation Contract

A sequence representation should state how raw symbols become model input:

$$
z_{1:T'}
=
\operatorname{tokenize}(x_{1:T};\rho)
$$

where $\rho$ defines alphabet, vocabulary, segmentation, truncation, padding, masking, and special tokens. $T'$ may differ from the raw length $T$.

| Domain | Token Unit | Position Meaning | Main Risk |
| --- | --- | --- | --- |
| text | byte, character, subword, word | document order | chunking and context-window truncation |
| protein | amino acid, residue token, domain segment | residue index, chain, insertion code | homolog leakage and residue-index mismatch |
| genome | base, k-mer, region token, variant-centered window | coordinate system and strand | overlapping windows and assembly mismatch |
| time series | event, frame, measurement window | timestamp or sampling order | irregular sampling and future leakage |
| action trace | action, observation, tool call | execution order | hidden state and environment mismatch |

## Positional Information

Sequence models need a convention for position:

$$
h_t
=
E(x_t) + P(t)
$$

where $E$ is a token embedding and $P(t)$ is a positional encoding. Position may be absolute, relative, rotary, temporal, residue-indexed, or genomic-coordinate-based. The choice changes extrapolation and alignment behavior.

## Masking and Padding

For variable-length sequences, define valid positions:

$$
m_t \in \{0,1\}
$$

Losses and metrics should ignore padding:

$$
\mathcal{L}
=
\frac{\sum_t m_t\ell_t}{\sum_t m_t}
$$

This is critical for documents, proteins, genomes, audio frames, and batched action traces. Averaging per token and per sequence can support different claims.

## Split Risks

- Text: near-duplicate documents, copied passages, or shared source collections.
- Protein: homologous sequences, same family, same structure template, or MSA leakage.
- Genome: overlapping windows, same locus, same chromosome region, or annotation-source leakage.
- Time series: future information, overlapping windows, or subject/session leakage.
- Agent traces: same task template, hidden tool state, or evaluator leakage.

## Checks

- What does one token represent?
- Is position absolute, relative, periodic, temporal, or biologically indexed?
- Can the model see future positions?
- Is truncation removing the important signal?
- Are near-duplicate or homologous sequences split across train and test?
- Are padding and masked positions excluded from loss and metrics?
- Does the tokenization preserve the entity mapping needed for labels?

## Related

- [[concepts/modalities/text|Text]]
- [[entities/sequence|Sequence entity]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/learning/masked-modeling|Masked modeling]]
