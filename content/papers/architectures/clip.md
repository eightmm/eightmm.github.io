---
title: CLIP
aliases:
  - papers/clip
  - papers/learning-transferable-visual-models-from-natural-language-supervision
tags:
  - papers
  - architectures
  - multimodal
  - contrastive-learning
  - vision-language
---

# CLIP

> The paper trains an image encoder and a text encoder with contrastive supervision so natural language can act as a flexible visual classifier.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Learning Transferable Visual Models From Natural Language Supervision |
| Authors | Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever |
| Year | 2021 |
| Venue | ICML 2021 |
| arXiv | [2103.00020](https://arxiv.org/abs/2103.00020) |
| PMLR | [v139/radford21a](https://proceedings.mlr.press/v139/radford21a.html) |
| Official implementation | [openai/CLIP](https://github.com/openai/CLIP) |
| Status | full note started |

## One-Line Takeaway

CLIP is the canonical dual-encoder vision-language model: train image and text encoders to align matching image-text pairs, then use text prompts as zero-shot classifiers.

## Question

Classical supervised vision models learn a fixed label space:

$$
x_{\text{image}}
\rightarrow
y \in \{1,\dots,C\}.
$$

This creates several constraints:

- categories must be predefined;
- labels are expensive;
- new tasks require new data or heads;
- models learn dataset-specific concepts;
- natural language descriptions are discarded.

The web contains many image-text pairs:

$$
(I_i, T_i).
$$

The paper asks:

> Can natural language supervision train a visual model that transfers to many downstream tasks without task-specific labeled images?

The architecture answer is a contrastive dual encoder:

$$
I \xrightarrow{\text{image encoder}} z_I,
\qquad
T \xrightarrow{\text{text encoder}} z_T.
$$

## Main Claim

Learning to match images with their paired texts at large scale produces transferable visual representations.

For a batch of $N$ image-text pairs:

$$
\{(I_i,T_i)\}_{i=1}^{N},
$$

compute normalized embeddings:

$$
u_i = \frac{f_\theta(I_i)}{\lVert f_\theta(I_i)\rVert},
\qquad
v_j = \frac{g_\phi(T_j)}{\lVert g_\phi(T_j)\rVert}.
$$

Similarity:

$$
s_{ij}
=
\tau \, u_i^\top v_j,
$$

where $\tau$ is a learned or tuned logit scale.

The correct pairs are diagonal:

$$
i=j.
$$

Train with symmetric contrastive loss:

$$
L
=
\frac{1}{2}
\left(
L_{\text{image}\rightarrow\text{text}}
+
L_{\text{text}\rightarrow\text{image}}
\right).
$$

Image-to-text loss:

$$
L_{\text{image}\rightarrow\text{text}}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ij})}.
$$

Text-to-image loss:

$$
L_{\text{text}\rightarrow\text{image}}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ji})}.
$$

This turns every other example in the batch into an implicit negative.

## Architecture Contract

| Component | Role |
| --- | --- |
| image encoder | maps image to embedding |
| text encoder | maps text to embedding |
| projection heads | map modality features into shared space |
| normalization | makes dot product behave like cosine similarity |
| logit scale | controls contrastive softmax sharpness |
| symmetric contrastive loss | aligns paired images and texts |
| prompt templates | turn labels into text descriptions |

The model does not fuse image and text token-by-token during pretraining. It encodes each side separately and compares embeddings.

## Dual Encoder

CLIP uses two towers:

$$
z_I = f_\theta(I),
$$

$$
z_T = g_\phi(T).
$$

After projection and normalization:

$$
u = \operatorname{norm}(W_I z_I),
$$

$$
v = \operatorname{norm}(W_T z_T).
$$

The shared embedding space supports:

- image-to-text retrieval;
- text-to-image retrieval;
- zero-shot classification;
- embedding-based search;
- image-text similarity scoring.

The tradeoff is clear:

| Design | Strength | Weakness |
| --- | --- | --- |
| dual encoder | fast retrieval and classification over many texts/images | limited fine-grained cross-modal reasoning |
| cross encoder | rich token-level interaction | expensive for retrieval and large candidate sets |

CLIP chooses scalability.

## Image Encoder

The image encoder can be a CNN or a Vision Transformer-style model.

Abstractly:

$$
z_I = f_\theta(I).
$$

For a ResNet-like encoder, the model produces pooled visual features. For a ViT-like encoder:

$$
I
\rightarrow
\text{patch tokens}
\rightarrow
\text{Transformer}
\rightarrow
z_I.
$$

The paper is architecture-relevant because it shows the backbone can be trained from natural language supervision rather than fixed image labels.

## Text Encoder

The text encoder is a Transformer over tokenized text:

$$
T=(w_1,\dots,w_m).
$$

It outputs a text representation:

$$
z_T = g_\phi(w_{1:m}).
$$

Class labels are converted to prompts:

$$
\text{``a photo of a } c \text{''}.
$$

Then the text encoder embeds the prompt:

$$
v_c = g_\phi(\text{prompt}(c)).
$$

This is the key shift:

$$
\text{classifier weights}
\rightarrow
\text{text embeddings}.
$$

## Zero-Shot Classification

For a candidate class set:

$$
\mathcal{C}=\{c_1,\dots,c_K\},
$$

construct text prompts:

$$
T_k = \operatorname{prompt}(c_k).
$$

Encode:

$$
v_k = g_\phi(T_k).
$$

Given image embedding:

$$
u = f_\theta(I),
$$

score classes:

$$
p(y=k\mid I)
=
\frac{\exp(\tau u^\top v_k)}
{\sum_{\ell=1}^{K}\exp(\tau u^\top v_\ell)}.
$$

So a zero-shot classifier is synthesized by the text encoder.

This is why CLIP is architecture-significant: the classification head is replaced by language-conditioned class prototypes.

## Prompt Templates

The prompt is part of the classifier:

$$
v_c = g_\phi(\text{``a photo of a } c \text{''}).
$$

Different templates can change performance:

$$
\text{``a photo of a dog''}
\neq
\text{``a blurry photo of a dog''}
\neq
\text{``a sketch of a dog''}.
$$

Prompt ensembling averages or combines multiple text embeddings:

$$
v_c
=
\frac{1}{M}
\sum_{m=1}^{M}
g_\phi(\operatorname{prompt}_m(c)).
$$

This means the interface is flexible but not free of design choices.

## Contrastive Learning View

The batch similarity matrix is:

$$
S = \tau UV^\top,
$$

where:

$$
U,V \in \mathbb{R}^{N\times d}.
$$

The diagonal contains positive pairs:

$$
S_{ii}.
$$

Off-diagonal entries are negatives:

$$
S_{ij}, \quad i\neq j.
$$

The loss pushes:

$$
u_i^\top v_i
>
u_i^\top v_j
\quad
\text{for } j\neq i.
$$

This is similar to InfoNCE-style contrastive learning, but applied across modalities.

## Why Batch Size Matters

Each batch provides negatives. With batch size $N$, each image sees:

$$
N-1
$$

text negatives, and each text sees:

$$
N-1
$$

image negatives.

Larger batches make the contrastive task harder and more informative, but increase memory and distributed training complexity.

This is why CLIP is not only an architecture paper; it is also a scaling and data paper.

## Comparison to Supervised Image Classifier

| Property | Supervised Classifier | CLIP |
| --- | --- | --- |
| Supervision | fixed category labels | natural language paired with images |
| Output head | learned classifier matrix | text encoder produces class prototypes |
| Label space | fixed during training | open vocabulary via prompts |
| Transfer | fine-tuning often needed | zero-shot possible |
| Failure mode | dataset label bias | prompt/data/semantic bias |

A supervised classifier learns:

$$
P(y\mid I)
=
\operatorname{softmax}(W f(I)).
$$

CLIP learns:

$$
P(T\mid I)
\propto
\exp(f(I)^\top g(T)).
$$

## Comparison to Cross-Attention Vision-Language Models

| Property | CLIP Dual Encoder | Cross-Attention VLM |
| --- | --- | --- |
| Image-text interaction | final embedding dot product | token-level fusion |
| Retrieval speed | high | lower |
| Zero-shot classification | natural fit | possible but heavier |
| Fine-grained reasoning | limited | stronger |
| Pretraining objective | contrastive matching | matching, captioning, MLM, VQA-style objectives |

CLIP is ideal when many image-text comparisons must be scored quickly. It is weaker when the task needs detailed grounding or multi-step visual reasoning.

## Comparison to Vision Transformer

[[papers/architectures/vision-transformer|ViT]] shows that image patches can be treated as Transformer tokens. CLIP shows that such visual encoders can be trained using natural language supervision.

| Property | ViT | CLIP |
| --- | --- | --- |
| Main object | image encoder architecture | vision-language dual encoder |
| Supervision | image labels in original ViT paper | image-text pairs |
| Output | class prediction | shared image-text embedding |
| Transfer | supervised/fine-tuned | zero-shot via prompts |

CLIP can use ViT as its image tower, but CLIP's core contribution is the multimodal contrastive training setup.

## Evidence Reading

The paper evaluates zero-shot transfer across many vision datasets. The important evidence is not that CLIP is best on every task, but that natural language supervision can produce broad visual transfer.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Image-text contrastive training learns transferable visual features | zero-shot and transfer benchmarks | language supervision is broadly useful | data scale and curation are central |
| Natural language can define classifiers | prompt-based ImageNet and other evaluations | class labels can become text prototypes | prompt templates affect performance |
| Robustness improves vs narrow supervised models | distribution shift/stress test comparisons | broad supervision helps transfer | robustness is uneven across tasks |
| Scaling matters | model/data scale comparisons | large noisy datasets can work | not separable from dataset composition |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | vision-language representation learning |
| Input unit | image-text pairs |
| Output unit | shared embedding similarity |
| Architecture family | dual encoder |
| Objective | symmetric contrastive image-text loss |
| Main comparison | supervised vision classifiers and transfer baselines |
| Key interface | text prompts as classifiers |
| Main scaling factor | large image-text dataset and batch contrast |
| Not the claim | detailed image-text reasoning or pixel-level grounding |

## Molecular and Scientific Modeling Reading

CLIP is not a molecular architecture, but its design pattern generalizes:

$$
\text{object encoder}
\leftrightarrow
\text{text encoder}
$$

with contrastive alignment.

Possible scientific analogues:

- molecule structure/image paired with textual assay descriptions;
- protein sequence/structure paired with function descriptions;
- microscopy images paired with captions or metadata;
- molecular graph paired with natural-language property descriptions;
- retrieval over papers, figures, structures, and annotations.

Important caveat:

$$
\text{semantic alignment}
\neq
\text{physical correctness}.
$$

For molecular or protein modeling, contrastive language alignment can help retrieval and annotation, but it does not replace graph, geometric, or mechanistic models for structure-sensitive prediction.

## Implementation Notes

### Normalize Embeddings

CLIP-style similarity usually uses normalized embeddings:

$$
u = \frac{z_I}{\lVert z_I\rVert},
\qquad
v = \frac{z_T}{\lVert z_T\rVert}.
$$

Without normalization, dot products can be dominated by embedding norms.

### Logit Scale

The scale $\tau$ controls softmax sharpness:

$$
s_{ij}=\tau u_i^\top v_j.
$$

Too low: weak contrast. Too high: overconfident matching and unstable gradients.

### Distributed Negatives

For multi-GPU training, negatives should often include examples across devices:

$$
N_{\text{effective}} = N_{\text{local}}\times N_{\text{devices}}.
$$

This requires communication of embeddings across devices.

### Prompt Evaluation

Report:

- prompt templates;
- whether prompt ensembling is used;
- class name mapping;
- text preprocessing;
- zero-shot candidate set.

Without this, zero-shot numbers are hard to compare.

## Failure Modes

### Prompt Sensitivity

The same class can have multiple natural language descriptions. Performance can vary with wording.

### Dataset Bias

Image-text pairs from the internet encode cultural, social, and distributional biases.

### Shortcut Matching

The model can learn correlations in captions and image styles rather than robust visual concepts.

### Poor Fine-Grained Grounding

Dual-encoder similarity may not localize which image region corresponds to which phrase.

### Semantic but Not Causal

CLIP learns association, not necessarily causal or physical understanding.

## Common Misreadings

### "CLIP understands images like a human."

No. It learns an image-text embedding space from large-scale paired data. It can transfer broadly, but it has biases and blind spots.

### "CLIP is a captioning model."

No. Base CLIP scores image-text similarity. It does not autoregressively generate captions.

### "Zero-shot means no design choices."

No. Prompt templates, class names, preprocessing, and candidate labels are design choices.

### "CLIP is just ViT."

No. CLIP can use ViT, but CLIP is the dual-encoder contrastive vision-language training framework.

## Later-Paper Checklist

When reading later vision-language or multimodal embedding papers, ask:

- Is the model dual encoder, cross encoder, or hybrid?
- What defines positive pairs?
- What negatives are used?
- Are embeddings normalized?
- Is the loss symmetric?
- How large is the batch/effective negative set?
- What prompt templates are used?
- Does evaluation test retrieval, classification, grounding, or reasoning?
- Are gains from architecture, data scale, filtering, or prompts?
- Can the model localize or only rank global embeddings?

## Why It Matters

CLIP is a key architecture paper because it made the dual-encoder contrastive pattern a standard interface for multimodal models:

$$
\text{encode image}
\cdot
\text{encode text}
\rightarrow
\text{semantic similarity}.
$$

It also changed classification:

$$
\text{fixed classifier head}
\rightarrow
\text{text-defined classifier}.
$$

This is a major bridge from vision backbones to multimodal foundation models.

## Limitations

- Requires very large paired image-text data.
- Prompt choice affects zero-shot behavior.
- Dual encoders are weak for fine-grained cross-modal reasoning.
- Internet-scale supervision introduces bias and noise.
- Similarity does not imply grounded explanation.
- The architecture does not encode physical, graph, or 3D structure by itself.

## Connections

- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/index|Architecture papers]]
