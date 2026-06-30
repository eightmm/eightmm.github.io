---
title: Layer Normalization
aliases:
  - papers/layer-normalization
  - papers/layernorm
tags:
  - papers
  - architectures
  - normalization
---

# Layer Normalization

> The paper introduced normalization over hidden units within a single example, avoiding dependence on mini-batch statistics.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Layer Normalization |
| Authors | Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton |
| Year | 2016 |
| Venue | arXiv preprint |
| arXiv | [1607.06450](https://arxiv.org/abs/1607.06450) |
| PDF | [author PDF](https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf) |
| Status | verified |

## Question

[[papers/architectures/batch-normalization|Batch Normalization]] made normalization a standard deep-learning component, but it depends on mini-batch statistics. That is awkward for recurrent networks, variable-length sequences, online inference, and settings where batch size changes between training and deployment.

Layer Normalization asks:

$$
\text{Can normalization stabilize hidden activations using only statistics from the current example?}
$$

The architectural issue is the axis over which statistics are computed. If the statistics use other examples in the mini-batch, the layer has batch-dependent behavior. If the statistics use hidden units within the current example, the layer can behave the same way in training and inference.

## Main Claim

LayerNorm normalizes across hidden units within a layer for each example, making the computation independent of mini-batch composition and consistent between training and inference.

For a hidden vector $a\in\mathbb{R}^{H}$:

$$
\mu = \frac{1}{H}\sum_{i=1}^{H} a_i
$$

$$
\sigma^2 =
\frac{1}{H}
\sum_{i=1}^{H}
(a_i-\mu)^2
$$

$$
\operatorname{LN}(a)_i
=
\gamma_i
\frac{a_i-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\beta_i.
$$

The learned parameters $\gamma_i$ and $\beta_i$ restore representational flexibility after normalization.

The durable architecture claim is:

$$
\text{per-example feature normalization}
\Rightarrow
\text{scale-stable hidden computation without batch-statistic state}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | hidden vector, token embedding, recurrent state, or residual-stream vector |
| Output | same shape, normalized over feature/hidden dimension |
| Statistics | mean and variance from the current example/token |
| Train-time behavior | uses current input statistics |
| Inference behavior | same formula as training |
| Learned parameters | usually per hidden feature $\gamma,\beta$ |
| State | no running mean or running variance |
| Inductive bias | stabilize feature scale inside each example |

For a sequence tensor $X\in\mathbb{R}^{B\times T\times d}$, Transformer-style LayerNorm usually treats each token independently:

$$
\mu_{bt}
=
\frac{1}{d}\sum_{j=1}^{d}X_{btj}
$$

$$
\sigma^2_{bt}
=
\frac{1}{d}\sum_{j=1}^{d}(X_{btj}-\mu_{bt})^2
$$

$$
Y_{btj}
=
\gamma_j
\frac{X_{btj}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}
+
\beta_j.
$$

The normalized axes are therefore the feature dimensions, while batch index $b$ and token index $t$ remain separate.

## Axis Contract

The main difference between normalization families is not the formula shape, but the axis:

| Method | Statistics Over | Keeps Separate | Train/Eval Difference |
| --- | --- | --- | --- |
| BatchNorm | batch and often spatial axes | channel/feature axis | yes, running stats at eval |
| LayerNorm | feature axis inside one example/token | batch and token positions | usually no |
| [[papers/architectures/root-mean-square-layer-normalization|RMSNorm]] | feature axis, RMS only | batch and token positions | usually no |
| GroupNorm | groups of channels | batch and spatial positions | usually no |
| InstanceNorm | spatial axes per example/channel | batch and channel | usually no |

Writing just "normalization" is incomplete. A correct architecture note should say which axis provides statistics and whether inference uses stored state.

## Method

LayerNorm is inserted as a differentiable layer:

$$
y = \operatorname{LN}(x)
$$

or inside a block:

$$
y = x + F(\operatorname{LN}(x)).
$$

The normalization can be applied to:

- recurrent hidden state;
- feed-forward hidden units;
- Transformer residual stream;
- token embeddings before attention or MLP blocks;
- final representation before a task head.

Unlike BatchNorm, LayerNorm does not require running estimates:

$$
\theta_{\text{LN}}=\{\gamma,\beta\}
$$

instead of:

$$
\theta_{\text{BN}}=\{\gamma,\beta,\hat{\mu},\hat{\sigma}^2\}.
$$

This makes it easier to use for autoregressive inference and variable batch sizes.

## Block View

| Component | Role | Architecture Implication |
| --- | --- | --- |
| Per-example mean | centers the hidden vector | removes common feature offset within one token/example |
| Per-example variance | controls hidden-vector scale | stabilizes feature magnitude |
| $\epsilon$ | numerical stability | important for low-variance vectors and mixed precision |
| $\gamma$ | learned scale | allows each feature to recover useful amplitude |
| $\beta$ | learned shift | allows each feature to recover useful offset |
| Placement | pre/post/sandwich/final norm | changes gradient path and depth stability |

LayerNorm is therefore not just a preprocessing step. It is part of the residual block contract.

## Evidence Reading

The original paper evaluated LayerNorm in settings where BatchNorm was less natural, especially recurrent neural networks.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| LayerNorm stabilizes hidden dynamics | recurrent-network experiments | per-example normalization helps sequence models | original scale is much smaller than modern LLMs |
| LayerNorm avoids batch dependence | method definition and experiments | behavior is not tied to mini-batch composition | feature-axis choice still affects representation |
| LayerNorm can speed training | comparisons to unnormalized baselines | hidden-scale control improves optimization | later recipes add residual design, warmup, Adam variants |
| LayerNorm fits online/recurrent use | no running batch statistics | consistent train/eval computation | not always best for CNN feature maps |

The paper should be read as an axis-change paper: normalize across hidden units of the same example rather than across examples.

## LayerNorm In Recurrent Networks

For a recurrent update:

$$
h_t = \phi(W_x x_t + W_h h_{t-1} + b)
$$

LayerNorm can normalize the pre-activation:

$$
h_t =
\phi(
\operatorname{LN}(W_x x_t + W_h h_{t-1})
).
$$

or normalize components inside gated recurrent units. For [[papers/architectures/long-short-term-memory|LSTM]], gate logits can be normalized before applying sigmoid/tanh:

$$
\begin{bmatrix}
i_t \\
f_t \\
o_t \\
\tilde{c}_t
\end{bmatrix}
=
\begin{bmatrix}
\sigma \\
\sigma \\
\sigma \\
\tanh
\end{bmatrix}
\left(
\operatorname{LN}(W x_t + U h_{t-1})
\right).
$$

The point is that recurrent models process one time step at a time, so batch statistics can be unstable or inconvenient. LayerNorm uses the current hidden vector, so it is more natural for causal, online computation.

## LayerNorm In Transformers

LayerNorm became central after [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]. A Transformer block is usually built from residual branches:

$$
x_{l+1}=x_l+F_l(x_l).
$$

Normalization placement determines whether the residual stream has a clean identity path.

Post-norm:

$$
x_{l+1}
=
\operatorname{LN}(x_l + F_l(x_l)).
$$

Pre-norm:

$$
x_{l+1}
=
x_l + F_l(\operatorname{LN}(x_l)).
$$

Pre-norm often improves depth stability because gradients can pass through the residual stream without always going through the normalization Jacobian. Post-norm normalizes the output of each residual addition, but can be harder to train at depth.

This means that a Transformer paper is underspecified if it only says "uses LayerNorm." It should say:

- pre-norm, post-norm, sandwich norm, or final norm;
- LayerNorm or RMSNorm;
- whether normalization has bias;
- whether normalization parameters receive weight decay;
- whether residual branches are scaled or gated.

## Pre-Norm vs Post-Norm

| Placement | Formula | Practical Reading |
| --- | --- | --- |
| Post-norm | $\operatorname{LN}(x+F(x))$ | original Transformer style; normalized block output |
| Pre-norm | $x+F(\operatorname{LN}(x))$ | common in deep modern sequence models; easier gradient path |
| Sandwich norm | $\operatorname{LN}(x+F(\operatorname{LN}(x)))$ | extra stabilization, extra compute |
| Final norm | $\operatorname{LN}(x_L)$ before head | normalizes final residual stream |

The placement can change training stability as much as many named architecture modifications. When comparing models, norm placement is part of the architecture.

## LayerNorm vs BatchNorm

| Dimension | LayerNorm | BatchNorm |
| --- | --- | --- |
| Statistics | hidden/features in one example/token | examples and often spatial positions |
| Train/eval formula | usually same | usually different |
| Running state | none | running mean and variance |
| Batch-size sensitivity | low | high |
| Common domain | Transformers, RNNs, sequence models | CNNs and residual vision models |
| Cross-example coupling | none through statistics | yes during training |
| Deployment risk | mostly numerical/placement | train/eval mode and stale stats |

LayerNorm is usually the safer default for sequence models because each token can be normalized independently of the batch.

## LayerNorm vs RMSNorm

[[papers/architectures/root-mean-square-layer-normalization|RMSNorm]] removes mean subtraction:

$$
\operatorname{RMSNorm}(x)
=
\gamma
\frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}.
$$

Compared with LayerNorm:

| Dimension | LayerNorm | RMSNorm |
| --- | --- | --- |
| Mean subtraction | yes | no |
| Scale normalization | variance/std | root mean square |
| Parameters | usually $\gamma,\beta$ | often only $\gamma$ |
| Cost | slightly higher | slightly lower |
| Common use | classic Transformers, BERT-era models | many modern decoder LLMs |

RMSNorm is not a different architecture family; it is a normalization variant that changes the centering and parameter contract.

## Gradient And Scale Intuition

LayerNorm makes the block less sensitive to absolute feature scale. If $x$ is multiplied by a positive scalar $a$, then:

$$
\operatorname{LN}(a x)
\approx
\operatorname{LN}(x)
$$

up to $\epsilon$ and affine parameters. This gives a form of scale invariance. But LayerNorm also changes gradients because the output depends on all features in the normalized vector.

The Jacobian is not diagonal: each output feature depends on the vector mean and variance. This couples feature dimensions within the same token/example. That coupling is useful for scale control, but it means LayerNorm is not just a per-feature rescaling.

## Implementation Notes

Important details to check:

| Detail | Why It Matters |
| --- | --- |
| normalized shape | defines exactly which axes are normalized |
| epsilon | affects numerical stability and low-variance behavior |
| affine parameters | $\gamma,\beta$ can be enabled or disabled |
| bias choice | some modern norms omit bias |
| dtype | variance computation can be sensitive in low precision |
| placement | pre/post/sandwich/final norm changes optimization |
| weight decay | norm parameters are often excluded |
| fused kernels | affect speed, memory, and numerical behavior |

For $X\in\mathbb{R}^{B\times T\times d}$, `LayerNorm(d)` normally normalizes the last dimension. But if tensors are laid out differently, the normalized shape can change the semantics.

## Common Misreadings

### "LayerNorm is just BatchNorm without batches"

Not quite. It changes the axis, removes running-stat state, and changes what information is coupled during normalization.

### "LayerNorm always improves models"

It often stabilizes training, but the best normalization depends on architecture, modality, depth, batch size, optimizer, and placement.

### "Pre-norm and post-norm are minor implementation details"

They change gradient paths through residual stacks. For deep Transformers, this is architecture-level behavior.

### "LayerNorm explains Transformer performance by itself"

LayerNorm is one component. Transformer performance also depends on attention, MLPs, residual connections, positional encoding, optimizer, data, scale, and training objective.

## What To Check In Architecture Papers

- Which normalization is used: LayerNorm, RMSNorm, BatchNorm, GroupNorm, or none?
- Which axes are normalized?
- Is the block pre-norm or post-norm?
- Is there a final norm before the output head?
- Are $\gamma,\beta$ present?
- Are norm parameters excluded from weight decay?
- Is $\epsilon$ reported?
- Does the baseline use the same normalization placement?
- Are claims about architecture separated from training-stability changes?
- Does the model use fused or approximate normalization kernels?

## Why It Still Matters

LayerNorm became one of the default building blocks for sequence modeling. It sits behind:

- [[papers/architectures/attention-is-all-you-need|Transformer]] encoder-decoder blocks;
- [[papers/architectures/bert|BERT]] encoder stacks;
- [[papers/architectures/gpt-2|GPT-2]] decoder-only stacks;
- [[papers/architectures/vision-transformer|Vision Transformer]] patch-token models;
- many modern [[concepts/architectures/state-space-model|state-space]] and hybrid sequence models.

For an architecture paper shelf, LayerNorm is essential because many later architecture claims are partly claims about stable residual streams.

## Limitations

- LayerNorm does not use batch statistics, which can be a disadvantage in some CNN regimes where batch/spatial aggregation is useful.
- It couples feature dimensions within each token/example.
- It can hide scale problems without solving deeper optimization or initialization issues.
- Its effect depends heavily on placement in residual blocks.
- The original paper predates large pre-norm Transformers, so modern conclusions require reading later work too.
- Low-precision implementations can differ numerically across kernels and frameworks.

## Connections

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/index|Architecture papers]]
