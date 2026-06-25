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

## Practical Checks

- Identify what information the encoder exposes to the decoder: pooled vector, token states, graph states, or pair states.
- Check whether the decoder can see previous outputs, ground-truth outputs, or only its own predictions.
- Look for train-test mismatch in autoregressive decoding and teacher-forced training.
- For molecule or protein tasks, check whether constraints are enforced during decoding or repaired afterward.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/learning/index|Learning methods]]
