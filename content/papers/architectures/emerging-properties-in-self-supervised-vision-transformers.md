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
| Status | full note |

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

## Projection head and output space

The student and teacher do not necessarily compare the raw class-token embedding. Each backbone can be followed by a projection head:

$$
z_s
=
h_s\left(f_{\theta_s}(v_s)\right),
\qquad
z_t
=
h_t\left(f_{\theta_t}(v_t)\right).
$$

The output is a vector of logits in a shared prototype space:

$$
z_s,z_t\in\mathbb{R}^{K}.
$$

The softmax distributions are:

$$
p_s(k\mid v_s)
=
\frac{
\exp(z_s^{(k)}/\tau_s)
}{
\sum_{j=1}^{K}
\exp(z_s^{(j)}/\tau_s)
},
$$

$$
p_t(k\mid v_t)
=
\frac{
\exp((z_t^{(k)}-c^{(k)})/\tau_t)
}{
\sum_{j=1}^{K}
\exp((z_t^{(j)}-c^{(j)})/\tau_t)
}.
$$

The projection head is discarded or bypassed for many downstream evaluations. This creates two representation interfaces:

| Interface | Used for |
| --- | --- |
| projection output | self-distillation target and optimization |
| backbone output | kNN, linear evaluation, dense transfer, or fine-tuning |

A result should state which interface is evaluated. High quality in the projection space does not automatically imply identical quality in the backbone feature.

## Multi-view cross-entropy

Let $\mathcal{V}_g$ be the set of global views and $\mathcal{V}_\ell$ the local views. The teacher usually processes only global views, while the student processes all views. A generalized loss is:

$$
\mathcal{L}
=
\frac{1}{|\mathcal{V}_s||\mathcal{V}_t|}
\sum_{v_s\in\mathcal{V}_s}
\sum_{v_t\in\mathcal{V}_t}
\mathbf{1}[v_s\ne v_t]\,
\mathcal{H}
\left(
p_t(\cdot\mid v_t),
p_s(\cdot\mid v_s)
\right),
$$

where:

$$
\mathcal{H}(p_t,p_s)
=
-\sum_{k=1}^{K}
p_t(k)\log p_s(k).
$$

The exclusion of identical view pairs is a protocol detail in common DINO implementations. It avoids spending all of the objective on trivial self-matching.

The training signal can be summarized as:

$$
\text{same image}
\Rightarrow
\text{same semantic distribution}
$$

across changes in crop, scale, color, and other augmentation factors.

This is different from pixel-level invariance. The student is not asked to reconstruct the pixels of a local crop. It is asked to predict a teacher distribution that summarizes the global image.

## Centering and sharpening

The teacher output can collapse when one prototype dimension dominates every image. DINO centers teacher logits using a running statistic:

$$
c_t
=
m c_{t-1}
+
(1-m)\,
\frac{1}{B}
\sum_{i=1}^{B}z_{t,i}.
$$

The exact aggregation can be distributed across workers, but the contract is a moving estimate of teacher output statistics.

Centering shifts the teacher distribution:

$$
z_t\rightarrow z_t-c_t.
$$

Sharpening uses a lower teacher temperature:

$$
\tau_t<\tau_s
$$

so that the teacher distribution is more concentrated. The two mechanisms have different effects:

| Mechanism | Main effect |
| --- | --- |
| centering | prevents persistent dominance by one output dimension |
| sharpening | makes the teacher target more selective |
| momentum | slows target movement and reduces feedback instability |

Collapse diagnostics should inspect entropy and marginal prototype usage rather than only the loss value.

For batch-averaged teacher marginal:

$$
\bar p_t(k)
=
\frac{1}{B}
\sum_{i=1}^{B}p_t(k\mid x_i),
$$

track:

$$
\mathcal{H}(\bar p_t)
=
-\sum_k\bar p_t(k)\log\bar p_t(k).
$$

A low marginal entropy can indicate that the teacher uses only a few output dimensions, though entropy alone is not a complete collapse test.

## Momentum teacher dynamics

The teacher is updated after the student optimization step:

$$
\theta_t^{(q+1)}
=
\lambda_q\theta_t^{(q)}
+
(1-\lambda_q)\theta_s^{(q+1)}.
$$

The momentum coefficient can be scheduled over training. A cosine schedule may move it toward one:

$$
\lambda_q
=
1-
(1-\lambda_{\mathrm{base}})
\frac{
\cos(\pi q/Q)+1
}{2},
$$

for a suitable training horizon $Q$ and base coefficient.

The teacher is not updated by the self-distillation gradient:

$$
\nabla_{\theta_t}\mathcal{L}=0
$$

for the teacher branch during the student update. The teacher changes through the EMA rule. Accidentally backpropagating through the teacher changes the optimization problem and can destabilize training.

The teacher is a low-pass filtered student:

$$
\theta_t
\approx
\operatorname{EMA}(\theta_s).
$$

This creates a slowly evolving target without an external pretrained checkpoint.

## Why view prediction is non-trivial

Global and local crops have different information content:

$$
\operatorname{Info}(v_{\mathrm{local}})
\subseteq
\operatorname{Info}(v_{\mathrm{global}})
$$

in the usual crop construction. The local student must infer a distribution compatible with a teacher that sees more of the image.

This creates a pressure toward object-level and context-robust features:

$$
\text{local patch evidence}
\rightarrow
\text{global semantic prototype}.
$$

It can also create shortcuts if augmentations are weak or if the data contains consistent backgrounds. The augmentation policy therefore defines the invariances that the model is asked to learn.

## Augmentation contract

The typical view generator includes transformations such as:

| Augmentation | Intended pressure |
| --- | --- |
| random resized crop | scale and partial-view invariance |
| horizontal flip | reflection invariance where valid |
| color jitter | color and illumination robustness |
| grayscale | reduced dependence on color |
| Gaussian blur | frequency robustness |
| solarization | stronger appearance variation on selected views |

For scientific images or biological structures, copying these transforms blindly can destroy the label-free identity relation. An augmentation is valid only if:

$$
\text{semantic identity}(x)
=
\text{semantic identity}(\mathcal{A}(x))
$$

under the intended task.

For a molecular graph, random atom deletion may not preserve identity; for a protein structure, arbitrary coordinate noise may break stereochemistry; for a microscopy image, cropping may remove the object of interest. DINO's objective is generic, but its view contract is domain-specific.

## Backbone interaction

DINO can use ViTs or convolutional backbones, but the paper highlights a synergy with ViT. The interaction can be decomposed:

| Component | ViT contribution | DINO contribution |
| --- | --- | --- |
| input unit | patch token | multi-view consistency over patch-token representations |
| global context | self-attention | teacher target can summarize full-image context |
| dense behavior | class-to-patch attention | self-supervision yields object-localized attention patterns |
| scaling | depth, width, patch size | momentum teacher and multi-crop training |

The main architectural lesson is not that DINO replaces ViT. It shows that the learning signal can expose properties of the backbone that are less visible under supervised training.

## Attention map extraction

For a ViT, let $A_{\ell,h}$ be the attention matrix at layer $\ell$ and head $h$. The class-token attention to patch tokens is:

$$
a_{\ell,h}
=
A_{\ell,h}[0,1:N+1].
$$

To aggregate heads:

$$
\bar a_\ell
=
\frac{1}{H}
\sum_{h=1}^{H}a_{\ell,h}.
$$

The vector can be reshaped from length $N$ to the patch grid and upsampled to image resolution. Choices matter:

- which layer is used;
- whether heads are averaged or selected;
- whether attention is averaged before or after normalization;
- whether residual attention is included;
- how the map is thresholded for segmentation metrics.

The attention map is evidence of a representation behavior, not a complete explanation of the classifier's decision.

## Evaluation layers

DINO uses multiple evaluation interfaces:

### kNN evaluation

Store normalized features from a reference set:

$$
\tilde z_i
=
\frac{z_i}{\|z_i\|_2}.
$$

For a query $z_q$, cosine similarity is:

$$
s(q,i)
=
\tilde z_q^\top\tilde z_i.
$$

The kNN prediction aggregates labels of the nearest reference features. This tests whether the geometry is useful without fitting a new linear classifier.

### Linear evaluation

Freeze the backbone and fit a classifier:

$$
\hat y
=
\operatorname{softmax}(Wz+b).
$$

This tests linearly accessible information, not the quality of the frozen feature under every downstream head.

### Fine-tuning

Update the backbone and head jointly. This tests initialization quality under adaptation, not frozen representation quality.

### Dense transfer

Use patch-level or spatial features for detection or segmentation. This is particularly relevant to the paper's object-localization observation.

These evaluations answer different questions and should not be combined into one “representation quality” number.

## What the paper establishes

The arXiv abstract reports that DINO-trained ViT features show explicit semantic segmentation information in attention maps, strong kNN behavior, and strong ImageNet linear evaluation. It also emphasizes the importance of momentum encoders, multi-crop training, and small patches.

The reported ImageNet linear evaluation result for ViT-Base is 80.1% top-1 in the cited setup. The result supports a specific package:

$$
\text{ViT}
+
\text{DINO objective}
+
\text{training recipe}
\rightarrow
\text{strong frozen representation}.
$$

It does not establish that the DINO objective alone causes the gain independent of backbone, data, augmentation, or compute.

## Ablation questions

- What changes when the teacher is not an EMA of the student?
- How much do centering and sharpening contribute separately?
- What is the effect of removing local crops while keeping global crops?
- How does patch size change dense transfer and attention localization?
- Does DINO remain stable with a convolutional backbone under the same view policy?
- Does kNN performance predict linear or dense transfer performance?
- How sensitive are results to teacher temperature and momentum schedule?
- Does the projection head improve backbone features or only stabilize its own output space?
- Which attention layer and head aggregation method best supports localization?
- Do domain-preserving augmentations produce the same emergent behavior outside natural images?

These questions distinguish architecture, target dynamics, view generation, and evaluation protocol.

## Relation to other self-supervised methods

| Method | Target | Main architectural pressure |
| --- | --- | --- |
| DINO | EMA teacher distribution | view-invariant semantic representation |
| MAE | masked input reconstruction | visible-token encoder and lightweight decoder |
| SimCLR | augmented-view contrastive logits | large negative set and projection geometry |
| BYOL | EMA target representation | bootstrap without explicit negatives |
| iBOT | teacher targets on masked patch tokens | patch-level self-distillation |

DINO's important boundary is teacher-student distillation without labels and without explicit negative pairs. It should be routed to [[ai/learning-methods|Learning Methods]] for the objective taxonomy, while its ViT interaction remains in Architecture Papers.

## Failure modes

### Representation collapse

All views produce nearly the same output:

$$
p_s(\cdot\mid v)
\approx
p_t(\cdot\mid v')
\approx
\text{constant}.
$$

Monitor feature variance, prototype usage, marginal entropy, and downstream performance.

### Teacher instability

If the teacher follows the student too quickly, the target can move faster than the student can learn. If it moves too slowly, adaptation may lag behind the data distribution.

### View shortcut

The student may predict crop or augmentation artifacts instead of semantic content. This is especially likely when local and global views have systematic differences.

### Attention over-interpretation

Attention maps may correlate with object regions while not being sufficient causal explanations. Use perturbation or dense evaluation when making stronger interpretability claims.

### Evaluation leakage

Using labels during model selection, augmentation tuning, or feature normalization can turn a nominally self-supervised comparison into a partially supervised one.

## Computational biology transfer

The teacher-student contract can be useful for sequences, structures, and molecules:

$$
\text{object}
\xrightarrow{\text{two valid views}}
(v_1,v_2)
\xrightarrow{\text{shared backbone}}
(z_1,z_2)
\xrightarrow{\text{EMA target}}
\text{invariant representation}.
$$

But the view generator must respect the entity and geometry:

| Object | Possible view | Guardrail |
| --- | --- | --- |
| protein sequence | masking, cropping, homolog-aware perturbation | avoid family leakage and invalid biological edits |
| protein structure | coordinate noise, residue masking, rigid transform | preserve or explicitly test E(3) symmetry |
| molecule graph | atom masking, bond-preserving augmentation | preserve valence and chemical validity |
| ligand pose | rigid-frame transform or local perturbation | distinguish physical pose variation from a new state |

For a geometric object, a rigid transform should satisfy the intended invariance/equivariance:

$$
f(RX+t)=f(X)
$$

for an invariant representation, or:

$$
f(RX+t)=Rf(X)+t
$$

for an equivariant coordinate output. A DINO-style loss does not enforce this automatically.

## Reproduction checklist

- [ ] State the backbone, patch size, projection-head dimension, and output prototypes.
- [ ] Record global/local crop counts, sizes, and augmentation probabilities.
- [ ] Specify student and teacher temperatures and the teacher momentum schedule.
- [ ] State the centering update, distributed aggregation, and initialization.
- [ ] Ensure teacher gradients are disabled and EMA update order is explicit.
- [ ] Report kNN, linear probe, fine-tuning, and dense metrics separately.
- [ ] Record feature layer, normalization, and attention-map extraction procedure.
- [ ] Monitor feature variance, prototype usage, entropy, and collapse indicators.
- [ ] Match data, training steps, batch size, and compute when comparing backbones.
- [ ] For biological data, define valid views, split policy, leakage controls, and symmetry contract.

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
