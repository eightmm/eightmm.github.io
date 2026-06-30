---
title: Emerging Properties in Self-Supervised Vision Transformers
aliases:
  - papers/dino
  - papers/emerging-properties-in-self-supervised-vision-transformers
tags:
  - papers
  - architectures
  - vision
  - transformer
  - self-supervised-learning
  - distillation
---

# Emerging Properties in Self-Supervised Vision Transformers

> DINO shows that self-supervised ViTs can learn strong visual representations and attention maps with semantic object structure through a teacher-student self-distillation setup without labels.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Emerging Properties in Self-Supervised Vision Transformers |
| Method | DINO, self-distillation with no labels |
| Authors | Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin |
| Year | 2021 |
| Venue | ICCV 2021 |
| arXiv | [2104.14294](https://arxiv.org/abs/2104.14294) |
| CVF | [ICCV 2021 open access](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) |
| Status | seed note |

## One-Line Takeaway

DINO is a canonical vision SSL paper for reading the interaction:

$$
\text{ViT architecture}
+
\text{self-distillation objective}
+
\text{multi-crop views}
\rightarrow
\text{strong representations and semantic attention maps}.
$$

It is not just a training trick. The paper is important because the emergent behavior is tied to Vision Transformers.

## Question

ViT showed that images can be represented as patch tokens:

$$
I
\rightarrow
\{x_1,\ldots,x_N\}
\rightarrow
\operatorname{ViTEncoder}.
$$

DINO asks whether self-supervised learning gives ViTs properties that differ from supervised ViTs or convolutional networks.

The concrete questions are:

1. Can a ViT learn useful representations without labels?
2. Do self-supervised ViT attention maps reveal object-level structure?
3. Which components make this training stable and useful?

## Architecture-Learning Contract

| Item | Contract |
| --- | --- |
| Backbone | Vision Transformer, also compared with convnets |
| Training signal | self-distillation without labels |
| Networks | student and momentum teacher |
| Views | multi-crop augmentations of the same image |
| Target | teacher probability distribution |
| Student input | global and local crops |
| Teacher input | global crops |
| Collapse control | centering, sharpening, teacher momentum |
| Evaluation | kNN, linear evaluation, attention/segmentation behavior |

This belongs in the architecture shelf because the paper's claim is not only "self-distillation works." It is that the ViT backbone exhibits useful emergent properties under this SSL setup.

## Teacher-Student Setup

DINO uses two networks with the same architecture:

$$
g_{\theta_s}
\quad\text{student}
$$

and:

$$
g_{\theta_t}
\quad\text{teacher}.
$$

The teacher is an exponential moving average of the student:

$$
\theta_t
\leftarrow
\lambda \theta_t
+
(1-\lambda)\theta_s.
$$

This is not a supervised teacher with labels. The teacher is a slowly moving target network.

For two augmented views $v_s$ and $v_t$ of the same image:

$$
p_s
=
\operatorname{softmax}
\left(
\frac{g_{\theta_s}(v_s)}{\tau_s}
\right),
$$

$$
p_t
=
\operatorname{softmax}
\left(
\frac{g_{\theta_t}(v_t)-c}{\tau_t}
\right),
$$

where:

| Symbol | Meaning |
| --- | --- |
| $\tau_s$ | student temperature |
| $\tau_t$ | teacher temperature |
| $c$ | center term used to stabilize teacher outputs |

The student is trained to match the teacher distribution:

$$
\mathcal{L}_{\text{DINO}}
=
-
\sum_k
p_t^{(k)}
\log p_s^{(k)}.
$$

This is cross-entropy between teacher and student outputs.

## Multi-Crop View Contract

DINO uses multiple augmented crops from the same image:

$$
\{v_1,\ldots,v_m\}
\sim
\mathcal{A}(x).
$$

The teacher sees global views. The student sees both global and local views.

| View | Given To | Role |
| --- | --- | --- |
| global crop | teacher and student | object-level target and matching |
| local crop | student | forces local-to-global consistency |

The useful pressure is:

$$
\text{local evidence}
\rightarrow
\text{match global teacher semantics}.
$$

This makes augmentation policy part of the method, not a preprocessing footnote.

## Collapse Control

Teacher-student SSL can collapse if all images map to the same output:

$$
p_s(x) = p_t(x) = \text{constant}.
$$

DINO uses several mechanisms to avoid this:

| Mechanism | Role |
| --- | --- |
| teacher momentum | provides a slower target |
| centering | prevents one dimension from dominating |
| sharpening | keeps teacher targets informative |
| multi-crop | creates view-consistency pressure |

The architecture note should record these because representation collapse is a method-level failure mode that can make the backbone look worse or better for the wrong reason.

## Relation to ViT

The paper's distinctive observation is that self-supervised ViT attention maps can align with object regions.

In a ViT, class-token attention to patch tokens can be inspected:

$$
a_j
=
\operatorname{Attention}_{\text{cls}\rightarrow j}.
$$

DINO-trained ViTs often show attention maps where high-attention patches correspond to salient objects.

The claim is not:

$$
\text{attention is always explanation}.
$$

The narrower reading is:

$$
\text{DINO + ViT}
\rightarrow
\text{attention maps with useful object-localization behavior}.
$$

That makes the paper a good bridge between [[papers/architectures/vision-transformer|ViT]], [[concepts/learning/self-supervised-learning|self-supervised learning]], and representation evaluation.

## Relation to MAE

| Paper | SSL Route | Architecture Pressure |
| --- | --- | --- |
| [MAE](/papers/architectures/masked-autoencoders-are-scalable-vision-learners) | masked patch reconstruction | visible-only encoder plus lightweight decoder |
| DINO | teacher-student self-distillation | ViT representations aligned across views |

Both are ViT-era vision SSL papers, but they ask different questions:

$$
\text{MAE: reconstruct missing input}
$$

$$
\text{DINO: match teacher representation across views}
$$

For this wiki, MAE lives near autoencoding and masked modeling; DINO lives near distillation, SSL, and representation evaluation.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| DINO features work well for kNN and linear evaluation | ImageNet representation evaluation | label-free ViT features are useful | metric depends on evaluation protocol |
| self-supervised ViT attention maps reveal object structure | qualitative and segmentation-style analysis | emergent localization can appear without labels | attention maps are not causal explanations |
| momentum teacher matters | ablations | stable target network improves SSL | exact schedule and temperature matter |
| multi-crop training matters | ablations | local/global view matching improves representation | augmentation semantics can fail in other domains |
| small patches help ViTs | backbone comparisons | finer patch tokens improve dense visual behavior | patch size increases sequence length and compute |

## Implementation Reading

Check:

- backbone: ViT-S, ViT-B, or convnet baseline;
- patch size, especially small-patch ViT settings;
- global crop count and local crop count;
- teacher momentum schedule;
- student and teacher temperatures;
- centering update rule;
- projection head dimension and output dimension;
- whether evaluation is kNN, linear probe, full fine-tuning, or dense transfer;
- whether reported attention maps come from class-token attention or another extraction rule.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "DINO is supervised distillation." | The teacher is a momentum version of the student; no human labels are used. |
| "DINO is an architecture block like attention." | It is an SSL method whose important findings depend strongly on ViT architecture. |
| "Attention maps prove explanation." | They are useful localization signals, not proof of causal importance. |
| "DINO and MAE are the same because both are SSL." | DINO matches teacher outputs across views; MAE reconstructs masked patches. |
| "Good kNN accuracy proves all downstream usefulness." | Representation evaluation depends on protocol, dataset, and transfer target. |

## What to Remember

DINO should be remembered as:

$$
\text{self-distillation}
+
\text{multi-crop views}
+
\text{momentum teacher}
+
\text{ViT}
\rightarrow
\text{strong SSL representation}.
$$

The architecture-level lesson is:

$$
\text{backbone properties can emerge only under the right learning signal}.
$$

That matters for this wiki because many architecture papers are really architecture-objective-data packages. DINO is a clean example of that interaction.

## Links

- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/masked-autoencoders-are-scalable-vision-learners|MAE]]
- [[papers/architectures/clip|CLIP]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[papers/architectures/index|Architecture papers]]
