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
| Status | full paper note |

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

## Activation Placement Contract

The location of nonlinearities is part of the block definition. A common MobileNetV2 implementation uses a bounded ReLU-style activation after the expansion and depthwise operations, but no activation after the final projection:

$$
x
\xrightarrow{1\times1}
\operatorname{ReLU6}
\xrightarrow{\text{depthwise}}
\operatorname{ReLU6}
\xrightarrow{1\times1}
y.
$$

The final output is then either added to the shortcut or passed to the next block. Replacing the final linear projection with ReLU6 changes the information path at the narrow interface:

$$
\operatorname{ReLU6}(W_{\text{proj}}h)
\ne
W_{\text{proj}}h.
$$

When reproducing MobileNetV2, record activation type, clipping range, normalization order, and whether the projection output is activated. “Uses an inverted residual block” does not specify these details completely.

## Tensor Shape Walkthrough

Let:

$$
x\in\mathbb{R}^{B\times H\times W\times C_{\mathrm{in}}}.
$$

For an expansion factor $t$, stride $s$, and output width $C_{\mathrm{out}}$:

| Step | Tensor shape | Operation |
| --- | --- | --- |
| input bottleneck | $B\times H\times W\times C_{\mathrm{in}}$ | thin residual representation |
| expansion | $B\times H\times W\times tC_{\mathrm{in}}$ | pointwise channel mixing plus nonlinearity |
| spatial filtering | $B\times H'\times W'\times tC_{\mathrm{in}}$ | depthwise convolution with stride $s$ |
| projection | $B\times H'\times W'\times C_{\mathrm{out}}$ | linear pointwise projection |
| output | same as projection | add shortcut only when shape matches |

The spatial dimensions satisfy approximately:

$$
H'\approx\frac{H}{s},
\qquad
W'\approx\frac{W}{s},
$$

subject to padding and rounding conventions. The residual condition is:

$$
\text{use }x+y
\quad\Longleftrightarrow\quad
s=1
\;\land\;
C_{\mathrm{in}}=C_{\mathrm{out}}.
$$

At a stage transition, the block still uses the expand-depthwise-project path but omits the identity addition because the spatial or channel shape changes.

## Why Expand Before Spatial Filtering

The depthwise convolution applies one spatial filter per channel, so the expanded width determines how many independently filtered feature channels are available:

$$
tC
\rightarrow
\{\text{spatially filtered channels}}_{1}^{tC}
\rightarrow
C'.
$$

If depthwise filtering were applied directly in the narrow bottleneck, the block would have fewer channels in which to represent distinct local patterns. Expansion creates a richer intermediate feature space while keeping the residual interface and spatial operation relatively efficient.

The tradeoff is explicit in the compute expression:

$$
C_{\text{block}}
\approx
HWtC^2
+
HWK^2tC
+
HWtCC'.
$$

Increasing $t$ improves internal capacity but raises both pointwise costs linearly. The depthwise term remains cheap relative to dense spatial convolution, but the pointwise projections can become the actual bottleneck.

## Linear Bottlenecks and Information Geometry

The paper's intuition is often described using a low-dimensional manifold. Suppose useful features lie near a lower-dimensional set embedded in the expanded representation space. A nonlinear map in the expanded space can transform the manifold, but an activation applied after projection to a narrow space can collapse distinct points:

$$
u\ne v
\quad\text{but}\quad
\operatorname{ReLU}(W u)
=
\operatorname{ReLU}(W v).
$$

Once two inputs map to the same bottleneck state, a later deterministic block cannot recover the lost distinction. The linear projection does not guarantee information preservation, but it removes one avoidable source of rank and sign collapse.

This should be stated as design intuition rather than a universal theorem:

$$
\text{expanded nonlinear transform}
\rightarrow
\text{linear narrow interface}
$$

is a favorable default when the narrow state is intended to carry a reusable residual representation.

## Block Family and Stage Parameters

A MobileNetV2 network is defined by a sequence of block specifications rather than one repeated block:

$$
\mathcal{B}
=
\{(t_i,c_i,n_i,s_i)\}_{i=1}^{S},
$$

where $t_i$ is expansion ratio, $c_i$ output channels, $n_i$ repeat count, and $s_i$ the stride of the first block in a stage.

| Parameter | Effect |
| --- | --- |
| expansion ratio $t$ | internal feature capacity and pointwise cost |
| output width $c$ | residual interface width and next-stage input |
| repeat count $n$ | sequential depth at a spatial scale |
| first-block stride $s$ | spatial downsampling and output stride |
| input resolution | activation area at every stage |

This representation makes the architecture reproducible and supports controlled variants. Changing only one stage's expansion ratio is a different experiment from applying a global width multiplier.

## Residual Stream and Memory

The narrow residual path reduces the size of activations that must be carried between blocks:

$$
\text{residual storage}
\propto
HW C_{\text{bottleneck}}
$$

while the expanded activation exists only inside the block:

$$
\text{temporary expansion storage}
\propto
HW(tC_{\text{bottleneck}}).
$$

The peak memory still depends on implementation and operator scheduling. A compiler may fuse operations or materialize intermediate tensors differently. The architectural intent is to keep long-lived block interfaces thin, not to guarantee a fixed memory footprint under every runtime.

## Comparison with a Classic Bottleneck

The two blocks differ in where the wide representation lives:

| Property | Classic residual bottleneck | MobileNetV2 inverted residual |
| --- | --- | --- |
| skip representation | usually wider stage representation | thin bottleneck representation |
| internal transform | compressed relative to input stage width | expanded relative to input bottleneck |
| spatial convolution | often dense | depthwise |
| final activation | may be nonlinear depending on variant | linear at narrow projection |
| main concern | optimize deep dense CNNs | minimize mobile cost while preserving local capacity |

The adjective “inverted” refers to this width placement, not to reversing the order of all operations.

## Detection and Segmentation Interface

MobileNetV2 was evaluated beyond classification, including lightweight detection and segmentation routes. The backbone/head boundary matters:

| Task | Backbone output requirement | Additional concern |
| --- | --- | --- |
| classification | global semantic feature | final pooling and classifier cost |
| detection | multi-scale or selected feature maps | output stride and localization detail |
| segmentation | spatial feature maps and decoder interface | preserving resolution and boundary information |

An efficient backbone can look strong on classification while losing small-object or boundary performance after downsampling. For dense prediction, record the feature pyramid or decoder and do not attribute the complete system result to the MobileNetV2 block alone.

## Evidence and Claim Boundaries

The paper reports accuracy/operation/parameter tradeoffs and demonstrates mobile models for classification, object detection, and semantic segmentation. Read the evidence at the level of the claim:

| Claim | Evidence | Boundary |
| --- | --- | --- |
| inverted residual is an effective block | controlled block and network comparisons | not a proof of universal optimality |
| linear bottlenecks preserve useful capacity | activation-placement ablations and intuition | not a theorem for every width or activation |
| model is efficient | MAdd and parameter comparisons | MAdd is not identical to latency |
| block transfers to detection/segmentation | SSDLite and mobile DeepLab-style systems | head/decoder contributes to the result |
| architecture scales across model sizes | multiple width/resource operating points | data and training recipe also scale |

The strongest durable contribution is the block contract. Benchmark superiority depends on hardware, task, input resolution, and training configuration.

## Ablation Matrix

| Ablation | Isolates | Expected diagnostic |
| --- | --- | --- |
| final activation versus linear projection | value of the linear bottleneck | narrow output with activation may lose information |
| expansion ratio $t$ | internal capacity versus pointwise cost | too small underfits; too large wastes compute |
| depthwise versus dense spatial convolution | spatial factorization benefit | separates operator efficiency from width placement |
| inverted versus classic bottleneck | residual interface geometry | tests where the wide representation should live |
| stride-1 versus stride-2 block | downsampling behavior | distinguishes block quality from resolution loss |
| width and resolution scaling | global resource controls | reveals task-specific sensitivity |
| classification versus dense prediction | transfer behavior | exposes output-stride and localization limits |
| theoretical MAdd versus measured latency | systems validity | detects memory-bound or unsupported kernels |

The key ablation is not only “ReLU or no ReLU.” It is the interaction among projection width, expansion ratio, nonlinearity placement, and residual addition.

## Implementation Pitfalls

1. **Activating the projection**: adding ReLU6 after the narrow projection changes the linear bottleneck contract.
2. **Adding a shortcut on shape mismatch**: residual addition requires equal spatial and channel shapes.
3. **Using the wrong stride location**: stride in the depthwise operation changes output resolution and aliasing.
4. **Confusing width and expansion**: a width multiplier changes stage interfaces; $t$ changes temporary internal width.
5. **Counting only depthwise cost**: pointwise convolutions often dominate MAdd and memory traffic.
6. **Comparing classification heads**: a large head can hide or reverse backbone efficiency.
7. **Ignoring output stride**: dense prediction depends on where downsampling occurs.
8. **Treating MAdd as latency**: runtime support, precision, and kernel fusion determine actual speed.
9. **Changing normalization silently**: batch statistics and ordering affect small-batch mobile training.

For a minimal block test:

1. verify the shape after expansion and projection;
2. verify no activation is applied after the linear projection;
3. verify the shortcut is used only for matching shapes;
4. compare gradients with and without the final activation;
5. benchmark pointwise and depthwise operations separately.

## Relation to EfficientNet and NAS

MobileNetV2 supplies a reusable block, while later systems choose how to scale and arrange such blocks:

| Family | Reuses or changes |
| --- | --- |
| MBConv | keeps inverted bottleneck, depthwise convolution, and squeeze/activation variants |
| EfficientNet | combines mobile blocks with compound depth/width/resolution scaling |
| neural architecture search | searches stage repeats, widths, kernel sizes, and expansion ratios |
| hardware-aware NAS | optimizes measured latency or energy rather than only MAdd |
| modern mobile ViTs | replaces or mixes convolutional blocks with token mixers/attention |

The general design pattern is:

$$
\text{thin interface}
\rightarrow
\text{wide cheap transform}
\rightarrow
\text{thin linear interface}
$$

This pattern can be evaluated independently of the exact MobileNetV2 stage table.

## Transfer to Scientific Workloads

The inverted residual idea can be useful for image-like scientific data when a narrow representation must be preserved across many blocks but local feature computation needs temporary capacity. However, the same domain checks as MobileNetV1 still apply:

- channel mixing must respect the meaning of modalities or feature types;
- spatial downsampling must preserve the scale of the scientific signal;
- vector/tensor channels require representation-aware operations;
- graph or molecular inputs need permutation/geometric guarantees that ordinary convolution does not provide;
- measured latency and memory must reflect the actual scientific pipeline.

The block can be a resource-efficient encoder component, but its use in computational biology requires a separate argument about entities, symmetries, and target resolution.

## Reproduction Checklist

- [ ] record stage tuples $(t,c,n,s)$ and input resolution;
- [ ] verify expand-depthwise-project order;
- [ ] verify activation placement, especially the linear projection;
- [ ] verify shortcut conditions for stride and channel equality;
- [ ] record kernel size, padding, normalization, and precision;
- [ ] calculate pointwise and depthwise costs separately;
- [ ] compare expansion ratio and width multiplier as different variables;
- [ ] report MAdd, parameters, peak memory, batch-1 latency, and throughput;
- [ ] evaluate classification and dense prediction interfaces separately;
- [ ] run a tiny forward/gradient test before training;
- [ ] compare against MobileNetV1 and a dense residual baseline under matched budgets.

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
