---
title: Captioning
tags:
  - tasks
  - multimodal
  - generation
---

# Captioning

Captioning generates text conditioned on a non-text input such as an image, video, molecule, structure, table, or plot. It is a conditional generation task.

The conditional language model is:

$$
p(y_{1:T}\mid x)=\prod_{t=1}^{T}p(y_t\mid y_{<t}, h_x)
$$

where $h_x$ is an encoded representation of the input.

## Common Pattern

1. Encode the non-text input.
2. Expose the encoded states to a text decoder.
3. Generate a sequence with an autoregressive or iterative decoder.
4. Evaluate both language quality and grounding.

## Checks

- Is the caption grounded in the input or filled with dataset priors?
- Does the metric reward factuality, coverage, fluency, or exact wording?
- Are captions descriptive, explanatory, diagnostic, or instructional?
- Does the model hallucinate objects, events, or experimental claims?
- Is the decoder allowed to copy labels from metadata?

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/llm/language-model|Language model]]
