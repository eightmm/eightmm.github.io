---
title: Encoder-Decoder Architectures
tags:
  - architectures
  - encoder-decoder
  - sequence-modeling
---

# Encoder-Decoder Architectures

Encoder-decoder architectures map an input representation into an output representation through separate encoding and generation or prediction stages.

The basic decomposition is:

$$
h = \operatorname{Encoder}(x),
\qquad
\hat{y} = \operatorname{Decoder}(h)
$$

For autoregressive decoders, the output distribution is usually factorized as:

$$
p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, h)
$$

## Key Ideas

- The encoder builds a representation of the input; the decoder consumes that representation to produce outputs.
- Decoders may be autoregressive, parallel, iterative, or task-specific prediction heads.
- Cross-attention is the usual bridge in Transformer encoder-decoder systems.
- Bottlenecks, latent variables, or pooled states force compression; full cross-attention preserves token-level access.
- The same pattern appears in translation, captioning, conditional generation, docking-style prediction, and masked reconstruction.

## Interface Contract

| Component | Input | Output | Common Failure |
| --- | --- | --- | --- |
| Encoder | source tokens, graph, image patches, residues, atoms | hidden states $H$ | representation drops needed information |
| Bridge | pooled vector, latent variable, cross-attention, pair state | conditioning interface | hidden bottleneck or leakage |
| Decoder | previous outputs plus context | output tokens, coordinates, graph, label | exposure bias or invalid decoding |
| Head | decoder state or pooled state | task-specific prediction | metric mismatch |

Write the interface explicitly:

$$
H = E_\theta(x),
\qquad
s_t = D_\phi(y_{<t}, H),
\qquad
p_\phi(y_t\mid y_{<t},x)=\operatorname{softmax}(W_o s_t)
$$

For non-autoregressive decoders:

$$
\hat{Y}=D_\phi(H, \epsilon, c)
$$

where $\epsilon$ can be noise or a latent seed and $c$ can be constraints.

## Training vs Inference

| Phase | Decoder Input | Risk |
| --- | --- | --- |
| teacher forcing | ground-truth previous tokens $y_{<t}$ | train-test mismatch |
| autoregressive inference | model samples $\hat{y}_{<t}$ | error accumulation |
| parallel decoding | mask, noise, slots, or queries | output alignment and duplicate predictions |
| iterative refinement | previous prediction | stopping rule and compute budget |
| constrained decoding | grammar, valence, geometry, or schema constraints | post-hoc repair can hide failures |

The paper should state whether decoding failures are counted or filtered away.

## Domain Examples

| Domain | Encoder | Decoder | Extra Constraint |
| --- | --- | --- | --- |
| translation | source text | target text | causal mask and tokenization |
| image captioning | image patches | text tokens | multimodal alignment |
| molecule generation | graph/string/latent | SMILES, graph, conformer | chemical validity |
| protein modeling | sequence, MSA, structure | sequence, coordinates, contacts | residue indexing and homolog leakage |
| docking | protein pocket and ligand | pose, score, interaction graph | receptor state and pose quality |
| agents | task state, memory, tools | action or tool call | action schema and verification |

## Practical Checks

- Identify what information the encoder exposes to the decoder: pooled vector, token states, graph states, or pair states.
- Check whether the decoder can see previous outputs, ground-truth outputs, or only its own predictions.
- Look for train-test mismatch in autoregressive decoding and teacher-forced training.
- For molecule or protein tasks, check whether constraints are enforced during decoding or repaired afterward.
- Is the encoder output fixed, retrieved, generated, or updated during decoding?
- Are decoding compute budget, beam/search policy, and invalid outputs reported?

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/learning/index|Learning methods]]
