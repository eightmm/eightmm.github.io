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

## Grounding Boundary

The output is text, but the claim is not only linguistic:

$$
\hat{y}
=
\operatorname{Decode}(h_x),
\qquad
\operatorname{Grounded}(\hat{y},x)=1
$$

where $\operatorname{Grounded}$ means the caption can be supported by the input. A fluent caption can still be wrong if it names objects, events, molecular properties, or experimental conclusions that are not visible or provided.

## Training Target

Supervised captioning often uses teacher-forced next-token loss:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T}
\log p_\theta(y_t\mid y_{<t},h_x)
$$

This optimizes likelihood of reference wording, not necessarily complete grounding or factual coverage. Multiple valid captions can exist for the same input.

## Evaluation

Caption evaluation should separate:

- fluency: is the text well formed?
- coverage: does it mention the important content?
- grounding: is every concrete claim supported by the input?
- utility: does the caption help the downstream user or task?
- safety: does it avoid unsupported private or experimental claims?

For scientific captions, exact wording metrics are usually weaker than evidence-grounded checks and human or tool-assisted review.

## Checks

- Is the caption grounded in the input or filled with dataset priors?
- Does the metric reward factuality, coverage, fluency, or exact wording?
- Are captions descriptive, explanatory, diagnostic, or instructional?
- Does the model hallucinate objects, events, or experimental claims?
- Is the decoder allowed to copy labels from metadata?
- Are multiple valid captions possible for the same input?
- Are unsupported claims counted as failures even if the text is fluent?
- Is the caption evaluated against the input, not only against a reference sentence?

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/llm/language-model|Language model]]
