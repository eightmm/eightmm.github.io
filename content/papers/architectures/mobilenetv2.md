---
title: MobileNetV2
aliases:
  - papers/mobilenetv2
  - papers/mobilenet-v2
  - papers/inverted-residuals-and-linear-bottlenecks
tags:
  - papers
  - architectures
  - cnn
  - vision
  - efficient-models
---

# MobileNetV2

> The paper made inverted residuals and linear bottlenecks a standard building block for efficient CNN backbones.

## Metadata

| Field | Value |
| --- | --- |
| Paper | MobileNetV2: Inverted Residuals and Linear Bottlenecks |
| Authors | Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen |
| Year | 2018 |
| Venue | CVPR 2018 |
| arXiv | [1801.04381](https://arxiv.org/abs/1801.04381) |
| CVF | [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) |
| Status | verified |

## Question

Large CNNs can perform well, but mobile and embedded settings care about multiply-adds, memory access, activation size, and latency. The question is not only how to reduce parameter count. The architecture must preserve representational capacity while keeping the expensive spatial operation cheap.

The paper asks:

$$
\text{Can a CNN block be both expressive and cheap enough for mobile vision?}
$$

The durable answer is the inverted residual block with a linear bottleneck:

$$
\text{thin}
\rightarrow
\text{expand}
\rightarrow
\text{cheap spatial filtering}
\rightarrow
\text{linear projection}
\rightarrow
\text{thin residual output}.
$$

## Main Claim

MobileNetV2 proposes a CNN block that inverts the classic residual bottleneck. Instead of doing expensive spatial convolution in a narrow hidden representation after compression, it expands channels, applies depthwise spatial convolution, and then projects back to a narrow output without a nonlinearity.

The core claim is:

$$
\text{inverted residual}
+
\text{linear bottleneck}
+
\text{depthwise separable convolution}
\Rightarrow
\text{better accuracy/efficiency tradeoff for mobile CNNs}.
$$

This is an architecture-block paper. The contribution is not just a smaller network; it is a reusable block design that influenced many later efficient vision models.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image feature map $X\in\mathbb{R}^{H\times W\times C}$ |
| Output | image feature map or class logits after stacked blocks |
| Main block | inverted residual with linear bottleneck |
| Spatial operation | depthwise convolution in expanded channel space |
| Channel mixing | pointwise $1\times1$ convolutions |
| Residual path | connects thin bottleneck input/output when shape matches |
| Efficiency target | mobile and resource-constrained vision |
| Downstream tasks | classification, detection through SSDLite, segmentation through mobile DeepLabv3 variant |

## Standard Residual Bottleneck vs MobileNetV2

Classic residual bottlenecks usually compress, process, and expand:

$$
C
\rightarrow
C/r
\rightarrow
C/r
\rightarrow
C'
$$

MobileNetV2 reverses the intuition for efficient blocks:

$$
C
\rightarrow
tC
\rightarrow
tC
\rightarrow
C'
$$

where $t$ is the expansion factor.

| Block Type | Hidden Width | Spatial Conv | Residual Stream |
| --- | --- | --- | --- |
| classic bottleneck | narrow middle | often dense conv | wide input/output |
| MobileNetV2 inverted residual | wide middle | depthwise conv | narrow input/output |

The residual stream stays narrow, which reduces memory and the cost of feature maps that must be kept across layers. The internal transformation is expanded, which gives the block room to compute.

## Depthwise Separable Convolution

A standard convolution maps:

$$
X\in\mathbb{R}^{H\times W\times C_{\mathrm{in}}}
\rightarrow
Y\in\mathbb{R}^{H\times W\times C_{\mathrm{out}}}
$$

with kernel size $K\times K$:

$$
Y_{u,v,c_{\mathrm{out}}}
=
\sum_{\Delta u,\Delta v,c_{\mathrm{in}}}
W_{\Delta u,\Delta v,c_{\mathrm{in}},c_{\mathrm{out}}}
X_{u+\Delta u,v+\Delta v,c_{\mathrm{in}}}.
$$

Its approximate compute is:

$$
H W K^2 C_{\mathrm{in}} C_{\mathrm{out}}.
$$

Depthwise separable convolution factorizes this into:

1. a depthwise spatial convolution per channel;
2. a pointwise $1\times1$ convolution that mixes channels.

Depthwise step:

$$
Z_{u,v,c}
=
\sum_{\Delta u,\Delta v}
D_{\Delta u,\Delta v,c}
X_{u+\Delta u,v+\Delta v,c}.
$$

Pointwise step:

$$
Y_{u,v,c_{\mathrm{out}}}
=
\sum_c
P_{c,c_{\mathrm{out}}}
Z_{u,v,c}.
$$

Approximate compute becomes:

$$
H W K^2 C_{\mathrm{in}}
+
H W C_{\mathrm{in}} C_{\mathrm{out}}.
$$

Compared with dense convolution:

$$
\frac{
H W K^2 C_{\mathrm{in}} + H W C_{\mathrm{in}} C_{\mathrm{out}}
}{
H W K^2 C_{\mathrm{in}} C_{\mathrm{out}}
}
=
\frac{1}{C_{\mathrm{out}}}
+
\frac{1}{K^2}.
$$

For $K=3$ and large $C_{\mathrm{out}}$, the spatial filtering cost is much smaller than dense convolution.

## Inverted Residual Block

Let the input be:

$$
x\in\mathbb{R}^{H\times W\times C}
$$

and let $t$ be the expansion ratio.

The block expands channels with a pointwise convolution:

$$
h_1
=
\sigma(W_{\mathrm{exp}} *_{1\times1} x),
\qquad
h_1\in\mathbb{R}^{H\times W\times tC}.
$$

It applies depthwise spatial filtering:

$$
h_2
=
\sigma(D *_{\mathrm{dw}} h_1).
$$

It projects back to a narrow bottleneck:

$$
y
=
W_{\mathrm{proj}} *_{1\times1} h_2.
$$

When stride is 1 and the input/output channels match, the block uses a residual connection:

$$
\operatorname{Block}(x)
=
x + y.
$$

The full pattern is:

$$
x
\xrightarrow{1\times1,\ \mathrm{expand}}
h_1
\xrightarrow{3\times3,\ \mathrm{depthwise}}
h_2
\xrightarrow{1\times1,\ \mathrm{linear\ project}}
y
\xrightarrow{+\ x}
z.
$$

## Linear Bottleneck

MobileNetV2 removes the nonlinearity after the final projection. The projection is linear:

$$
y
=
W_{\mathrm{proj}} h_2
$$

rather than:

$$
y
=
\sigma(W_{\mathrm{proj}} h_2).
$$

The paper's intuition is that narrow low-dimensional bottleneck layers should preserve information. Applying ReLU in a narrow space can collapse dimensions:

$$
\operatorname{ReLU}(z)_i=\max(z_i,0).
$$

If important variation crosses the zero boundary in a low-dimensional embedding, ReLU can destroy information that the next block cannot recover. MobileNetV2 therefore keeps nonlinear transformations in the expanded hidden space and uses a linear map at the narrow output.

The design rule is:

$$
\text{nonlinearity in high-dimensional expansion}
\quad
\text{but}
\quad
\text{linear narrow bottleneck output}.
$$

## Why the Residual Is Inverted

In a standard residual block, the skip connection often connects high-dimensional representations around a narrower transformation. In MobileNetV2, the skip connection connects bottlenecks:

$$
x_{\mathrm{thin}}
\rightarrow
\text{wide transform}
\rightarrow
y_{\mathrm{thin}}
$$

and the residual is:

$$
z_{\mathrm{thin}} = x_{\mathrm{thin}} + y_{\mathrm{thin}}.
$$

This is inverted relative to classic bottleneck thinking because the residual stream is low-dimensional while the internal transform is high-dimensional.

| Design Choice | Reason |
| --- | --- |
| thin residual stream | lower activation memory and cheaper block interfaces |
| expanded hidden layer | enough capacity for nonlinear transformation |
| depthwise spatial conv | cheap local filtering |
| linear projection | avoid information loss in narrow bottleneck |

## Block Complexity

For input width $C$, output width $C'$, expansion ratio $t$, and kernel size $K$, the MobileNetV2 block roughly costs:

Expansion:

$$
H W C(tC)
$$

Depthwise spatial filtering:

$$
H W K^2(tC)
$$

Projection:

$$
H W(tC)C'.
$$

Total:

$$
H W \left(tC^2 + K^2tC + tCC'\right).
$$

This explains the tradeoff:

- increasing $t$ raises expressive width and cost;
- depthwise convolution keeps spatial filtering cheap;
- pointwise convolutions still dominate when channels are large.

For efficient CNNs, the expensive operation is often not the $3\times3$ depthwise convolution but the $1\times1$ channel mixing.

## Relation to MobileNetV1

MobileNetV1 popularized depthwise separable convolutions for mobile vision. MobileNetV2 keeps that efficiency idea but changes the residual block structure.

| Axis | MobileNetV1 | MobileNetV2 |
| --- | --- | --- |
| core operation | depthwise separable convolution | inverted residual block |
| channel pattern | depthwise + pointwise | expand + depthwise + linear project |
| residual block | not the central idea | central design |
| bottleneck nonlinearity | less emphasized | remove nonlinearity at narrow output |

MobileNetV2 is the stronger architecture note because it defines the block that later mobile CNN families reuse, tune, or search over.

## Relation to EfficientNet

EfficientNet uses MBConv-style mobile inverted bottleneck blocks and then studies compound scaling. The dependency is:

$$
\text{MobileNetV2 block design}
\rightarrow
\text{MBConv family}
\rightarrow
\text{EfficientNet scaling}.
$$

So the two papers answer different questions:

| Paper | Main Question |
| --- | --- |
| MobileNetV2 | what efficient CNN block should be used? |
| EfficientNet | how should a strong CNN family be scaled? |

This is why MobileNetV2 belongs before EfficientNet in a vision-backbone reading path.

## Evidence to Read

The evidence should be read as an efficiency tradeoff, not just a top-1 accuracy comparison.

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet classification | the block is competitive as a backbone |
| multiply-add comparisons | the model is compute efficient by operation count |
| latency measurements | efficiency is not only parameter count |
| detection with SSDLite | the backbone transfers to object detection |
| segmentation with mobile DeepLabv3 variant | the backbone can support dense prediction |

The key reading question is:

$$
\text{Does the architectural block improve the accuracy-cost frontier?}
$$

not:

$$
\text{Does it maximize accuracy without constraints?}
$$

## What the Paper Changed

MobileNetV2 made several design ideas standard:

| Idea | Later Importance |
| --- | --- |
| inverted bottleneck | common in efficient CNNs and NAS search spaces |
| linear bottleneck | warns against careless nonlinearities in narrow spaces |
| depthwise separable spatial filtering | separates spatial filtering from channel mixing |
| mobile-aware evidence | encourages latency and operation-count reporting |
| backbone-plus-task evaluation | tests classification, detection, and segmentation routes |

The paper should be read as a bridge between classic CNN architecture and efficient foundation-backbone engineering.

## Failure Modes and Caveats

- Multiply-add count is not the same as wall-clock latency on every device.
- Depthwise convolution can be memory-bound or poorly optimized on some hardware.
- The block is designed around image/grid locality; it does not solve long-range interaction by itself.
- Expansion ratio, width multiplier, resolution, and implementation kernels affect the conclusion.
- The linear-bottleneck argument is architectural intuition plus empirical support, not a universal theorem.

## Reading Checks

When reading a later efficient CNN paper, ask:

| Question | Why It Matters |
| --- | --- |
| Is the block still an inverted residual? | many later models inherit MobileNetV2's block contract |
| Where are nonlinearities placed? | narrow-layer nonlinearities can change information flow |
| Are pointwise convolutions dominating cost? | depthwise conv may not be the bottleneck |
| Is latency measured on the target device? | FLOPs may mispredict deployment speed |
| Is the comparison at matched compute, params, and resolution? | otherwise architecture gain may be scaling gain |

## Links

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/efficientnet|EfficientNet]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/swin-transformer|Swin Transformer]]

## One-Line Memory

MobileNetV2 is the efficient CNN block paper: expand channels, do cheap depthwise spatial filtering, linearly project back to a thin bottleneck, and keep the residual path over the thin representation.
