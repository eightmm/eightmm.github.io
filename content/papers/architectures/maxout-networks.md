---
title: Maxout Networks
aliases:
  - papers/maxout
  - papers/maxout-networks
tags:
  - papers
  - architectures
  - activation-function
  - dropout
---

# Maxout Networks

> The paper introduced maxout units: learnable piecewise-linear activation blocks that take the maximum over multiple affine responses and pair naturally with dropout.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Maxout Networks |
| Authors | Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, Yoshua Bengio |
| Year | 2013 |
| Venue | ICML 2013 |
| arXiv | [1302.4389](https://arxiv.org/abs/1302.4389) |
| PMLR | [Proceedings page](https://proceedings.mlr.press/v28/goodfellow13.html) |
| Status | full note started |

## Question

Dropout randomly masks hidden units during training and can be interpreted as approximate model averaging. But not every activation function is equally suited to that training regime.

The paper asks:

$$
\text{Can we design a hidden unit that works especially well with dropout and learns its own activation shape?}
$$

Instead of fixing a scalar nonlinearity such as sigmoid, tanh, or ReLU, maxout lets the model choose the maximum among several learned affine pieces.

## Main Claim

A maxout unit computes:

$$
h_i(x)
=
\max_{j\in\{1,\ldots,k\}}
z_{i,j}(x)
$$

where each piece is affine:

$$
z_{i,j}(x)
=
x^\top W_{:,i,j}+b_{i,j}.
$$

So one unit is:

$$
h_i(x)
=
\max_j
(x^\top W_{:,i,j}+b_{i,j}).
$$

The architecture claim is:

$$
\text{max over learned affine pieces}
\Rightarrow
\text{learned convex piecewise-linear activation}.
$$

The training claim is:

$$
\text{maxout}
+
\text{dropout}
\Rightarrow
\text{strong regularized classifiers on vision benchmarks}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | vector, patch, or feature activation $x$ |
| Unit computation | maximum over $k$ affine projections |
| Output | one scalar per maxout unit |
| Hyperparameter | number of pieces $k$ |
| Nonlinearity | learned piecewise-linear convex function |
| Pairing | designed to work well with dropout |
| Main cost | more affine projections than a standard ReLU unit |

In tensor form, a linear layer that would normally produce:

$$
Z\in\mathbb{R}^{B\times d}
$$

instead produces:

$$
Z\in\mathbb{R}^{B\times d\times k}
$$

and reduces over the piece axis:

$$
H_{b,i}
=
\max_{j=1}^{k}
Z_{b,i,j}.
$$

This is why maxout is both an activation and a small architectural block.

## Relation To ReLU

ReLU is:

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

This can be viewed as a max over two fixed affine functions:

$$
\max(0,x).
$$

Maxout generalizes this:

$$
\max(w_1^\top x+b_1,\ldots,w_k^\top x+b_k).
$$

| Unit | Pieces | Learned Pieces? | Shape |
| --- | --- | --- | --- |
| ReLU | $0$ and $x$ | partly fixed | convex piecewise-linear |
| absolute value | $x$ and $-x$ | fixed | convex piecewise-linear |
| maxout | $k$ affine functions | yes | learned convex piecewise-linear |

This means maxout can implement ReLU-like behavior, absolute-value-like behavior, or a richer learned convex activation.

## Piecewise-Linear View

For one-dimensional intuition:

$$
h(x)=\max(a_1x+b_1,a_2x+b_2,\ldots,a_kx+b_k).
$$

The active piece is:

$$
j^\ast(x)
=
\arg\max_j (a_jx+b_j).
$$

The gradient follows the selected piece:

$$
\frac{dh}{dx}
=
a_{j^\ast(x)}
$$

except at boundaries where multiple pieces tie.

For high-dimensional inputs, each maxout unit partitions input space into regions where different affine pieces are active. This makes the unit behave like a learned local expert selector.

## Relation To Dropout

Dropout masks activations during training:

$$
\tilde{h}
=
m\odot h,
\qquad
m_i\sim\operatorname{Bernoulli}(p).
$$

Maxout was designed as a companion to this procedure. The paper argues that the max over affine pieces works well with dropout's approximate model averaging because the unit remains a simple piecewise-linear function under many masked subnetworks.

The useful reading distinction:

| Component | Role |
| --- | --- |
| dropout | regularizes by sampling subnetworks during training |
| maxout | provides flexible learned activation units |
| maxout + dropout | combines flexible units with strong regularized model averaging |

So maxout should not be read as just "another activation." It was proposed together with a training/regularization story.

## Relation To Network In Network

[[papers/architectures/network-in-network|Network In Network]] replaces local linear convolutional filters with local micro-networks. Maxout is another way to make local units more expressive:

| Direction | Mechanism |
| --- | --- |
| Maxout | one unit chooses among multiple affine pieces |
| Network In Network | each local patch is processed by an MLP-like micro-network |
| Inception | multiple local transformations are computed in parallel branches |

All three ask how to make local transformations richer than one linear filter plus one fixed nonlinearity.

## Capacity And Cost

A standard dense layer with output width $d$ uses:

$$
W\in\mathbb{R}^{d_{\text{in}}\times d}.
$$

A maxout layer with $k$ pieces uses:

$$
W\in\mathbb{R}^{d_{\text{in}}\times d\times k}.
$$

Parameter and compute cost scale roughly with $k$ before the max reduction.

The output width after maxout is still $d$:

$$
\mathbb{R}^{d\times k}
\rightarrow
\mathbb{R}^{d}.
$$

So maxout trades more candidate affine pieces for a fixed number of output units.

## Evidence To Read Carefully

The paper reports state-of-the-art classification performance at the time on MNIST, CIFAR-10, CIFAR-100, and SVHN using maxout with dropout.

For architecture reading, split the claims:

| Claim | Evidence Type | Caution |
| --- | --- | --- |
| Maxout works well with dropout | benchmark results with dropout-trained maxout networks | improvement depends on the regularization recipe |
| Learned activations are expressive | max over affine pieces can represent convex piecewise-linear functions | one maxout unit alone represents convex functions; richer functions require combinations |
| Maxout can compete with rectifier units | comparisons to rectifier-style networks | parameter count and hyperparameter tuning must be compared carefully |

The durable lesson is that activation choice can be an architectural capacity decision, not merely a scalar function choice.

## Failure Modes

| Failure Mode | Mechanism | Practical Check |
| --- | --- | --- |
| Parameter inflation | $k$ affine pieces per unit | compare matched parameter budgets |
| Over-crediting activation | gains may come from dropout, preprocessing, or model size | separate ablations |
| Non-sparse representation | maxout outputs are not inherently sparse like ReLU activations | inspect activation distributions |
| Piece collapse | some affine pieces may rarely win | monitor active-piece frequencies |
| Hardware inefficiency | extra projection dimension and max reduction may be less convenient than ReLU/GELU | check kernel and tensor layout |

## Where It Fits

| Axis | Placement |
| --- | --- |
| Architecture family | feed-forward activation/block |
| Core primitive | max over learned affine pieces |
| Main concept | [Activation function](/concepts/architectures/activation-function) |
| Regularization partner | [Dropout](/concepts/architectures/dropout) |
| Related local modeling | [Network In Network](/papers/architectures/network-in-network) |
| Later common alternatives | [GELU](/papers/architectures/gelu), [GLU variants](/papers/architectures/glu-variants-improve-transformer) |

## Practical Checks

When reading a paper that changes the activation or local block, ask:

| Question | Why It Matters |
| --- | --- |
| Is the activation fixed or learned? | learned activations add parameters and capacity |
| Does the comparison match parameter count? | maxout multiplies affine projections by $k$ |
| Is the result tied to dropout or another regularizer? | activation and training recipe can be inseparable |
| What is the active path selection rule? | maxout chooses a winning affine piece |
| Does the implementation report kernel cost? | simple equations can hide hardware cost |

## Related

- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/dropout|Dropout]]
- [[papers/architectures/network-in-network|Network In Network]]
- [[papers/architectures/gelu|GELU]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[papers/architectures/alexnet|AlexNet]]
