---
title: Flamingo
aliases:
  - papers/flamingo
  - papers/flamingo-visual-language-model
tags:
  - papers
  - architectures
  - multimodal
  - vision-language
  - few-shot-learning
  - transformer
---

# Flamingo

> Flamingo connects frozen vision and language backbones with a visual resampler and gated cross-attention so a language model can consume interleaved images, videos, and text in-context.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Flamingo: a Visual Language Model for Few-Shot Learning |
| Authors | Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan |
| Year | 2022 |
| Venue | NeurIPS 2022 |
| arXiv | [2204.14198](https://arxiv.org/abs/2204.14198) |
| Proceedings | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) |
| Status | seed note started |

## One-Line Takeaway

Flamingo is an early canonical visual language model architecture for few-shot multimodal prompting: keep strong unimodal models, compress visual inputs into a small token set, and let a causal language model attend to those visual tokens through gated cross-attention.

## Question

[[papers/architectures/clip|CLIP]] aligns images and text in a shared embedding space:

$$
z_I \approx z_T.
$$

That is useful for retrieval and zero-shot classification, but not enough for open-ended generation:

$$
(I, q)
\rightarrow
\text{text answer}.
$$

Large language models already perform in-context learning over text examples:

$$
(x_1,y_1,\ldots,x_k,y_k,x_\*)
\rightarrow
y_\*.
$$

Flamingo asks:

> Can a language model perform the same few-shot prompting pattern when the context contains images or videos interleaved with text?

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| frozen vision encoder | image or video frames | dense visual features | reusable visual representation |
| Perceiver resampler | visual features plus learned latents | fixed number of visual tokens | compress variable visual input |
| frozen language model | previous text tokens | causal hidden states | language generation prior |
| gated cross-attention layers | text hidden states query visual tokens | visual-conditioned text states | inject visual evidence without rewriting the LM |
| autoregressive objective | interleaved multimodal sequence | next text token loss | train the connector for prompted generation |

The model changes the VLM interface from:

$$
\operatorname{score}(I,T)
$$

to:

$$
p_\theta(y_{1:T} \mid I_{1:M}, x_{1:N}).
$$

## Perceiver Resampler

The vision encoder produces many visual features:

$$
X_v \in \mathbb{R}^{n_v \times d_v}.
$$

For images, $n_v$ can be the number of visual patches. For videos, it can grow with frames and patches. Directly cross-attending to all features at every language layer is expensive, so Flamingo uses a fixed set of learned latent queries:

$$
Z_0 \in \mathbb{R}^{m \times d},
\qquad
m \ll n_v.
$$

The resampler repeatedly updates the latent tokens by attending to visual features:

$$
Z_{\ell+1}
=
\operatorname{CrossAttn}
\left(
Q=Z_\ell,\ K=X_v,\ V=X_v
\right).
$$

The output:

$$
Z_v \in \mathbb{R}^{m \times d}
$$

is a compact visual memory. This is the key architecture move: variable-size image/video evidence becomes a fixed-size set of tokens that a language model can query.

## Gated Cross-Attention

A frozen causal language model maintains hidden states:

$$
H_\ell \in \mathbb{R}^{n_t \times d}.
$$

Flamingo inserts cross-attention blocks between LM layers. A simplified block is:

$$
\tilde{H}_\ell
=
H_\ell
+
\tanh(\alpha_\ell)
\operatorname{CrossAttn}
\left(
Q=\operatorname{LN}(H_\ell),\
K=Z_v,\
V=Z_v
\right),
$$

followed by the usual language-model feed-forward and self-attention updates.

The learned scalar or vector gate starts near zero:

$$
\tanh(\alpha_\ell) \approx 0.
$$

This makes multimodal training less destructive because the pretrained language pathway is initially preserved. As training proceeds, the model learns where visual evidence should influence text generation.

## Interleaved Multimodal Context

The input is not limited to one image and one caption. A prompt can be:

$$
(I_1, \text{text}_1, I_2, \text{text}_2,\ldots,I_q,\text{question})
$$

and the language model generates:

$$
p(y_t \mid y_{<t},\ I_{1:q},\ \text{text context}).
$$

This makes few-shot visual prompting possible:

| Prompt Part | Function |
| --- | --- |
| image-text examples | define task behavior |
| query image/video | provides new visual evidence |
| question or instruction | specifies output format |
| generated text | answer, caption, or multiple-choice response |

The architecture contribution is therefore not only a better image-text encoder. It defines a promptable multimodal sequence interface.

## Training Objective

Flamingo is trained autoregressively on text tokens conditioned on previous text and visual inputs:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T}
\log
p_\theta
\left(
y_t
\mid
y_{<t},\ I_{1:M},\ x_{1:N}
\right).
$$

Only text tokens are predicted. Images and videos provide conditioning context.

The important separation is:

$$
\text{vision encoder}
\rightarrow
\text{resampled visual tokens}
\rightarrow
\text{gated cross-attention into LM}.
$$

This is a reusable architecture pattern for adapting a language model to non-text evidence.

## Why It Matters

Flamingo marks a shift from contrastive image-text representation learning to generative multimodal foundation models.

| Before | Flamingo-style move |
| --- | --- |
| image-text similarity | text generation conditioned on visual input |
| fixed downstream heads | prompted few-shot task interface |
| one image-caption pair | arbitrarily interleaved visual and text context |
| end-to-end multimodal model from scratch | connect strong frozen unimodal backbones |

Later visual instruction models reuse the same broad idea:

$$
\text{frozen or pretrained vision encoder}
+
\text{small trainable connector}
+
\text{language model}.
$$

[[papers/architectures/blip-2|BLIP-2]] is a cleaner follow-up pattern with a Querying Transformer bridge.

## What To Watch

- The reported capability comes from architecture, web-scale multimodal data, prompt design, and model scale together.
- Gated cross-attention makes the LM visually conditioned, but it does not guarantee faithful visual grounding.
- Few-shot visual prompting can be sensitive to example order, answer format, and prompt wording.
- Frozen-backbone reuse improves efficiency, but the connector still needs enough data to learn alignment.
- Evaluation should distinguish captioning, VQA, retrieval, multiple-choice QA, and instruction following.

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/llm/language-model|Language model]]
- [[papers/architectures/clip|CLIP]]
- [[papers/architectures/blip-2|BLIP-2]]
- [[papers/architectures/gpt-3|GPT-3]]
