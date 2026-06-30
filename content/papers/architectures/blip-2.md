---
title: BLIP-2
aliases:
  - papers/blip-2
  - papers/bootstrapping-language-image-pre-training
tags:
  - papers
  - architectures
  - multimodal
  - vision-language
  - q-former
  - transformer
---

# BLIP-2

> BLIP-2 bridges a frozen image encoder and a frozen language model with a lightweight Querying Transformer, making multimodal pretraining more modular and parameter-efficient.

## Metadata

| Field | Value |
| --- | --- |
| Paper | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models |
| Authors | Junnan Li, Dongxu Li, Silvio Savarese, Steven C. H. Hoi |
| Year | 2023 |
| Venue | ICML 2023 |
| arXiv | [2301.12597](https://arxiv.org/abs/2301.12597) |
| PMLR | [v202/li23q](https://proceedings.mlr.press/v202/li23q.html) |
| Implementation | [salesforce/LAVIS](https://github.com/salesforce/LAVIS) |
| Status | seed note started |

## One-Line Takeaway

BLIP-2 makes vision-language modeling modular: freeze a strong image encoder, freeze a strong LLM, and train a compact Q-Former bridge that extracts text-relevant visual tokens for representation learning and generation.

## Question

End-to-end vision-language pretraining is expensive because it often updates large vision and language backbones together:

$$
\theta
=
\{\theta_{\text{vision}},\theta_{\text{text}},\theta_{\text{fusion}}\}.
$$

But strong unimodal models already exist:

$$
I \xrightarrow{f_{\text{img}}} X_v,
\qquad
T \xrightarrow{f_{\text{LM}}} H_t.
$$

BLIP-2 asks:

> Can a small trainable bridge exploit frozen pretrained vision encoders and frozen LLMs without retraining the whole multimodal stack?

## Architecture Contract

| Component | Trainable? | Input | Output | Role |
| --- | --- | --- | --- | --- |
| image encoder | frozen | image $I$ | visual features $X_v$ | strong visual representation |
| Q-Former | trainable | learned queries, text, $X_v$ | query embeddings $Z_q$ | extract language-relevant visual tokens |
| projection | trainable | $Z_q$ | LM-compatible embeddings | map bridge output into LLM space |
| LLM | frozen | visual prefix plus text tokens | generated text | language generation and instruction behavior |

The core interface is:

$$
I
\xrightarrow{\text{frozen image encoder}}
X_v
\xrightarrow{\text{Q-Former}}
Z_q
\xrightarrow{\text{projection}}
E_v
\xrightarrow{\text{frozen LLM}}
y_{1:T}.
$$

## Q-Former

The Querying Transformer contains learned query tokens:

$$
Q_0 \in \mathbb{R}^{m \times d}.
$$

The frozen image encoder returns visual features:

$$
X_v \in \mathbb{R}^{n \times d_v}.
$$

The Q-Former lets the learned queries attend to visual features:

$$
Z_q
=
\operatorname{QFormer}
\left(
Q_0,\ X_v,\ T
\right).
$$

A simplified cross-attention update is:

$$
Q_{\ell+1}
=
Q_\ell
+
\operatorname{CrossAttn}
\left(
Q=Q_\ell,\ K=X_v,\ V=X_v
\right).
$$

The bottleneck is intentional:

$$
m \ll n.
$$

Instead of sending every image patch to the LLM, BLIP-2 asks a small set of query tokens to retrieve the visual information that language needs.

## Two-Stage Pretraining

BLIP-2 trains the bridge in two stages.

| Stage | Frozen Modules | Trainable Module | Objective Role |
| --- | --- | --- | --- |
| vision-language representation learning | image encoder | Q-Former | align query output with text |
| vision-to-language generative learning | image encoder and LLM | Q-Former plus projection | make visual tokens interpretable by the LLM |

This separates two problems:

$$
\text{see relevant visual content}
\neq
\text{make an LLM generate from it}.
$$

## Stage 1: Representation Bridge

The first stage teaches query tokens to extract text-aligned visual evidence from a frozen image encoder.

A contrastive view is:

$$
s_{ij}
=
\operatorname{sim}
\left(
z^{(I)}_i,\ z^{(T)}_j
\right)/\tau.
$$

Then an image-to-text contrastive loss can be written as:

$$
\mathcal{L}_{\text{itc}}
=
-
\frac{1}{N}
\sum_i
\log
\frac{\exp(s_{ii})}
{\sum_j \exp(s_{ij})}.
$$

The paper also uses matching and language-modeling style objectives in the representation stage. The architecture point is that the Q-Former learns an image-text interface before it is attached to the LLM.

## Stage 2: Generative Bridge

The second stage connects Q-Former outputs to a frozen LLM. Query embeddings are projected into the LLM input space:

$$
E_v
=
W Z_q.
$$

The LLM then models text conditioned on this visual prefix:

$$
p(y_{1:T} \mid I, x)
=
\prod_{t=1}^{T}
p
\left(
y_t
\mid
y_{<t},\ E_v,\ x
\right).
$$

Training minimizes:

$$
\mathcal{L}_{\text{gen}}
=
-
\sum_t
\log
p
\left(
y_t
\mid
y_{<t},\ E_v,\ x
\right).
$$

The frozen LLM is not taught language from scratch. The bridge is taught to speak in a representation the LLM can use.

## BLIP-2 vs Flamingo

Both models connect pretrained visual and language components, but the connector differs.

| Axis | [[papers/architectures/flamingo|Flamingo]] | BLIP-2 |
| --- | --- | --- |
| Visual bridge | Perceiver resampler | Q-Former |
| LM access pattern | gated cross-attention inserted into LM layers | projected visual query tokens fed to frozen LLM |
| Main emphasis | interleaved few-shot multimodal prompting | modular compute-efficient VLP |
| Trainable path | multimodal connector and inserted attention | Q-Former and projection |
| Reusable lesson | condition an LM through visual cross-attention | learn a compact modality adapter |

The common pattern:

$$
\text{frozen image model}
+
\text{trainable connector}
+
\text{frozen or mostly frozen language model}.
$$

## Why It Matters

BLIP-2 is architecture-relevant because it made the connector explicit. It is not just a new dataset recipe.

| Design Choice | Architectural Consequence |
| --- | --- |
| frozen image encoder | reuse strong visual features and reduce training cost |
| frozen LLM | preserve language generation and instruction ability |
| Q-Former bottleneck | control visual token budget and learn task-relevant extraction |
| two-stage bridge training | separate alignment from generation |
| modular adapters | allow different image encoders and LLMs to be swapped |

This pattern is central for later multimodal assistants: the expensive foundation models can remain mostly fixed while a relatively small bridge learns cross-modal grounding.

## What To Watch

- Frozen components reduce trainable parameters but can limit domain adaptation.
- A small query bottleneck can discard details needed for dense visual reasoning.
- Text generation quality can hide visual grounding errors.
- Reported efficiency depends on which parameters are counted as trainable versus frozen.
- Image-text alignment and instruction-following generation are different claims and should be evaluated separately.

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/llm/language-model|Language model]]
- [[papers/architectures/clip|CLIP]]
- [[papers/architectures/flamingo|Flamingo]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
