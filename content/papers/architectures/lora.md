---
title: LoRA
aliases:
  - papers/lora
  - papers/low-rank-adaptation
  - papers/architectures/low-rank-adaptation
tags:
  - papers
  - architectures
  - fine-tuning
  - parameter-efficient-adaptation
---

# LoRA

> The paper introduced Low-Rank Adaptation, a parameter-efficient way to adapt pretrained models by freezing the base weights and training low-rank update matrices.

## Metadata

| Field | Value |
| --- | --- |
| Paper | LoRA: Low-Rank Adaptation of Large Language Models |
| Authors | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| Year | 2021 preprint; 2022 conference |
| Venue | ICLR 2022 |
| arXiv | [2106.09685](https://arxiv.org/abs/2106.09685) |
| OpenReview | [nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9) |
| Code | [microsoft/LoRA](https://github.com/microsoft/LoRA) |
| Status | full note started |

## One-Line Takeaway

LoRA freezes a pretrained weight matrix $W_0$ and learns a low-rank update:

$$
W
=
W_0
+
\Delta W,
\qquad
\Delta W
=
BA,
$$

where:

$$
B\in\mathbb{R}^{d_{\text{out}}\times r},
\qquad
A\in\mathbb{R}^{r\times d_{\text{in}}},
\qquad
r\ll \min(d_{\text{in}}, d_{\text{out}}).
$$

Only $A$ and $B$ are trained.

## Question

Full fine-tuning updates every parameter:

$$
\theta
\leftarrow
\theta+\Delta\theta.
$$

For a large pretrained model, this is expensive because each task may require storing and optimizing a full copy of the model.

The architecture question is:

$$
\text{Can downstream adaptation be represented by small low-rank changes inside large weight matrices?}
$$

LoRA answers by changing the adaptation path, not the pretrained base model.

## Main Claim

For large pretrained Transformers, downstream adaptation can be parameter-efficient when the weight updates are constrained to a low-rank subspace.

For a linear layer:

$$
h = W_0x,
$$

LoRA changes the forward pass to:

$$
h
=
W_0x
+
\frac{\alpha}{r}BAx.
$$

Here:

| Symbol | Meaning |
| --- | --- |
| $W_0$ | frozen pretrained weight |
| $A$ | trainable down projection |
| $B$ | trainable up projection |
| $r$ | rank |
| $\alpha$ | scaling hyperparameter |
| $\alpha/r$ | update scale |

The durable claim:

$$
\text{freeze base model}
+
\text{train low-rank residual paths}
\Rightarrow
\text{cheap task adaptation}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Base model | pretrained Transformer or other neural model |
| Frozen weights | original matrix $W_0$ is not updated |
| Trainable weights | low-rank matrices $A$ and $B$ |
| Insert location | usually attention projections and sometimes MLP projections |
| Forward form | $W_0x+\frac{\alpha}{r}BAx$ |
| Initialization | commonly $A$ random and $B=0$, so $\Delta W=0$ at start |
| Merge option | $W_0+\frac{\alpha}{r}BA$ can be merged for inference |
| Main benefit | far fewer trainable parameters and optimizer states |
| Main risk | rank/location choices limit adaptation capacity |

LoRA is not a new backbone. It is an adaptation block attached to existing backbone layers.

## Low-Rank Update

A full update to a matrix:

$$
\Delta W
\in
\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

has:

$$
d_{\text{out}}d_{\text{in}}
$$

parameters.

LoRA parameterizes the update as:

$$
\Delta W=BA
$$

with:

$$
B\in\mathbb{R}^{d_{\text{out}}\times r},
\qquad
A\in\mathbb{R}^{r\times d_{\text{in}}}.
$$

The trainable parameter count becomes:

$$
r(d_{\text{out}}+d_{\text{in}}).
$$

When $r$ is small:

$$
r(d_{\text{out}}+d_{\text{in}})
\ll
d_{\text{out}}d_{\text{in}}.
$$

This is the core compression.

## Forward Pass

For input:

$$
x\in\mathbb{R}^{d_{\text{in}}},
$$

the frozen path is:

$$
y_0=W_0x.
$$

The LoRA residual path is:

$$
y_{\text{LoRA}}
=
\frac{\alpha}{r}BAx.
$$

The output is:

$$
y
=
y_0+y_{\text{LoRA}}
=
W_0x+\frac{\alpha}{r}BAx.
$$

This is a residual adaptation:

$$
\text{base behavior}
+
\text{task-specific low-rank correction}.
$$

## Initialization

LoRA commonly initializes:

$$
B=0,
$$

and initializes $A$ randomly.

At step 0:

$$
\Delta W
=
BA
=
0.
$$

So the model starts exactly as the pretrained model:

$$
y=W_0x.
$$

This matters because the adapter path should not destroy pretrained behavior before learning begins.

## Where LoRA Is Inserted

In a Transformer attention block:

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V,
\qquad
O=\operatorname{Attn}(Q,K,V)W_O.
$$

LoRA can be applied to one or more projection matrices:

$$
W_Q,\ W_K,\ W_V,\ W_O.
$$

The important implementation question is:

$$
\text{Which matrices receive trainable low-rank updates?}
$$

Common choices include:

| Target | Effect |
| --- | --- |
| $W_Q,W_V$ | common efficient setting in the paper |
| $W_Q,W_K,W_V,W_O$ | larger adaptation capacity |
| MLP up/down projections | changes feed-forward transformation |
| embeddings or output head | task/domain specific, less universal |

## Merge For Inference

Because LoRA is linear, the update can be merged:

$$
W_{\text{merged}}
=
W_0+\frac{\alpha}{r}BA.
$$

Then inference uses:

$$
y=W_{\text{merged}}x.
$$

This avoids extra inference latency compared with adapter modules that add new nonlinear layers in the forward path.

However, merging has an operational consequence:

| Serving Mode | Tradeoff |
| --- | --- |
| merged weights | fastest for one task, task-specific checkpoint |
| separate adapters | flexible multi-task serving, extra adapter management |
| hot-swapped adapters | cheap storage per task, runtime routing complexity |

## Relation To Full Fine-Tuning

Full fine-tuning updates:

$$
W=W_0+\Delta W_{\text{full}},
$$

where $\Delta W_{\text{full}}$ can be full-rank.

LoRA constrains:

$$
\operatorname{rank}(\Delta W)
\le
r.
$$

This is a bias:

$$
\text{adaptation should live in a low-dimensional update subspace}.
$$

The bias is useful when:

- the pretrained model already contains most needed capability;
- the task update is small relative to the base model;
- memory and storage matter;
- multiple task-specific variants are needed.

It can fail when:

- the task is far from pretraining;
- rank is too low;
- LoRA is inserted into the wrong layers;
- data requires changing representations that the chosen matrices cannot affect enough.

## Relation To Adapters

Adapters usually add bottleneck modules:

$$
h'
=
h
+
W_{\text{up}}
\sigma
\left(
W_{\text{down}}h
\right).
$$

LoRA instead modifies existing linear maps with a low-rank residual:

$$
W_0x
\rightarrow
W_0x+\frac{\alpha}{r}BAx.
$$

| Axis | Adapter Modules | LoRA |
| --- | --- | --- |
| insertion | new modules between layers | low-rank update inside existing linear layers |
| nonlinearity | often yes | no, linear residual path |
| mergeability | usually not into one original matrix | mergeable into $W_0$ |
| inference latency | can add latency | can be zero after merge |
| capacity | depends on bottleneck and nonlinearity | depends on rank and target matrices |

This is why LoRA is often treated as an architecture component rather than only an optimizer trick.

## Relation To Linear Algebra

LoRA is directly connected to low-rank factorization.

If the adaptation update has low intrinsic rank:

$$
\Delta W
\approx
U_r\Sigma_rV_r^\top,
$$

then a factorized parameterization:

$$
\Delta W=BA
$$

can represent the important directions with fewer trainable parameters.

The key assumption:

$$
\text{downstream adaptation update is low-rank enough}.
$$

This is not guaranteed by the method. It is a modeling hypothesis that should be checked by rank sweeps and task performance.

## Evidence Pattern

The paper supports LoRA with:

| Evidence | What It Supports |
| --- | --- |
| parameter count comparison | trainable state is much smaller |
| memory comparison | optimizer state and gradients are reduced |
| quality on RoBERTa, DeBERTa, GPT-2, GPT-3 | low-rank updates can match strong baselines |
| rank analysis | adaptation appears rank-deficient in studied settings |
| latency comparison | merged LoRA avoids adapter-style inference overhead |

The important reading point is that LoRA's value is a three-way claim:

$$
\text{quality}
+
\text{trainable parameter efficiency}
+
\text{deployment practicality}.
$$

Any one metric alone is incomplete.

## Practical Reading Checks

| Question | Why |
| --- | --- |
| Which matrices receive LoRA? | target location controls adaptation capacity |
| What rank $r$ is used? | rank controls parameter count and expressivity |
| What is $\alpha$? | scaling changes update magnitude |
| Is dropout used on the LoRA path? | affects regularization |
| Are base weights frozen? | otherwise it is not pure LoRA adaptation |
| Are adapters merged at inference? | affects latency and checkpoint semantics |
| Are multiple adapters composed? | composition can change behavior unexpectedly |
| Is the baseline full fine-tuning or another PEFT method? | determines what claim is being made |

## Why It Belongs In Architecture Papers

LoRA is not a model family like Transformer or CNN, but it is a reusable architecture-level modification:

$$
\text{linear layer}
\rightarrow
\text{frozen linear layer plus low-rank trainable residual}.
$$

It belongs here because many modern papers assume:

- LoRA fine-tuning;
- QLoRA-style quantized base plus LoRA adapters;
- task-specific adapter merging;
- LoRA on attention projections;
- LoRA on diffusion U-Nets or vision models.

Without a LoRA note, later implementation and paper reviews become ambiguous about what actually changed in the model.

## Limits

- Low rank can underfit tasks requiring broad weight changes.
- Rank and target module choices are hyperparameters.
- LoRA does not reduce forward activation memory by itself.
- Frozen base model quality is still the main dependency.
- Merging adapters is simple for one task, but multi-adapter serving has operational complexity.
- LoRA can make evaluation misleading if adapter, base, and data provenance are not recorded.

The concise limitation:

$$
\text{LoRA reduces trainable update size}
\neq
\text{LoRA makes adaptation universally sufficient}.
$$

## What To Remember

- LoRA freezes pretrained weights and trains low-rank residual updates.
- The core equation is $W_0x+\frac{\alpha}{r}BAx$.
- Trainable parameter count is $r(d_{\text{out}}+d_{\text{in}})$ instead of $d_{\text{out}}d_{\text{in}}$.
- It is usually applied to attention projection matrices.
- The update can be merged into the base weight for inference.
- LoRA is an adaptation block, not a replacement for a strong pretrained model.

## Links

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/llama|LLaMA]]
