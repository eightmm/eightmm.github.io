---
title: ConvNeXt
aliases:
  - papers/convnext
  - papers/a-convnet-for-the-2020s
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# ConvNeXt

> The paper modernized a ResNet-style ConvNet using post-ViT design lessons and showed that pure convolutional backbones could remain competitive.

## Metadata

| Field | Value |
| --- | --- |
| Paper | A ConvNet for the 2020s |
| Authors | Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie |
| Year | 2022 |
| Venue | CVPR 2022 |
| arXiv | [2201.03545](https://arxiv.org/abs/2201.03545) |
| CVF | [CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) |
| Status | verified |

## Question

After [[papers/architectures/vision-transformer|Vision Transformer]] and [[papers/architectures/swin-transformer|Swin Transformer]], many strong vision results were attributed to Transformers. But Swin also reintroduced local windows, hierarchy, and dense-prediction-friendly feature maps. ConvNeXt asks a sharper question:

$$
\text{How much of the post-ViT gain comes from Transformer attention,}
$$

and how much comes from modern training recipes, stage design, large local kernels, normalization placement, and scaling choices?

The paper's strategy is:

$$
\text{ResNet baseline}
\rightarrow
\text{modernized ConvNet design}
\rightarrow
\text{Transformer-era comparison}.
$$

## Main Claim

ConvNeXt shows that a pure ConvNet, modernized with design choices learned from Transformer-era vision models, can compete with hierarchical Transformers on classification, detection, and segmentation while keeping the simplicity of convolutional modules.

The durable claim is not:

$$
\text{convolution always beats attention}.
$$

It is:

$$
\text{architecture comparison is confounded by training recipe and design modernization}.
$$

ConvNeXt is therefore both a model family and a controlled architecture reading exercise.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image tensor |
| Backbone type | pure convolutional hierarchical vision backbone |
| Starting point | ResNet-style ConvNet |
| Main modernization target | make ConvNet design resemble strong Transformer-era vision backbones where useful |
| Token mixing analogue | large-kernel depthwise convolution |
| Channel mixing | pointwise linear layers / $1\times1$ convolutions |
| Normalization | LayerNorm-style placement in ConvNet block |
| Activation pattern | fewer activations, GELU-style nonlinearity |
| Output use | classification, detection, segmentation backbone |

## Why It Belongs in Architecture Papers

ConvNeXt is not just a performance paper. It is an architecture audit of vision backbones:

$$
\text{ResNet}
\xrightarrow{\text{training + macro design + block design}}
\text{ConvNeXt}.
$$

It identifies which design changes matter when comparing ConvNets and Transformers. That makes it a good paper to read after:

1. [[papers/architectures/deep-residual-learning|Deep Residual Learning]];
2. [[papers/architectures/vision-transformer|Vision Transformer]];
3. [[papers/architectures/swin-transformer|Swin Transformer]].

## From ResNet Block to ConvNeXt Block

A simplified residual ConvNet block can be read as:

$$
y
=
x + F_{\mathrm{conv}}(x).
$$

ConvNeXt keeps this residual-block view but changes the branch $F$.

The ConvNeXt block can be summarized as:

$$
x
\xrightarrow{\mathrm{DWConv}_{7\times7}}
h
\xrightarrow{\mathrm{LayerNorm}}
\tilde h
\xrightarrow{\mathrm{Linear}\ 4C}
u
\xrightarrow{\mathrm{GELU}}
v
\xrightarrow{\mathrm{Linear}\ C}
F(x)
\xrightarrow{+\ x}
y.
$$

Here $\mathrm{DWConv}$ is depthwise convolution. It performs local spatial mixing independently per channel, while the pointwise linear layers mix channels.

This resembles the Transformer block separation:

$$
\text{token mixing}
\quad+\quad
\text{channel MLP}.
$$

But token mixing is still convolutional rather than attention-based.

## Depthwise Convolution as Local Token Mixing

For a feature map:

$$
X\in\mathbb{R}^{H\times W\times C},
$$

a depthwise convolution applies one spatial filter per channel:

$$
Z_{u,v,c}
=
\sum_{\Delta u,\Delta v}
D_{\Delta u,\Delta v,c}
X_{u+\Delta u,v+\Delta v,c}.
$$

ConvNeXt uses a larger spatial kernel than many classic CNN blocks:

$$
K=7.
$$

The point is not only receptive field size. A large depthwise kernel behaves like cheap local token mixing:

$$
\text{local window interaction}
\approx
\text{large-kernel depthwise convolution}.
$$

Compared with local self-attention, depthwise convolution uses fixed learned offsets rather than input-dependent pairwise weights.

| Mixing Type | Weight Pattern | Bias |
| --- | --- | --- |
| depthwise convolution | fixed offsets shared over positions | locality and translation sharing |
| window attention | content-dependent weights inside window | adaptive local interaction |
| global attention | content-dependent all-token weights | long-range interaction |

## Channel MLP Analogue

After spatial mixing, ConvNeXt uses pointwise channel mixing. In channel-last notation, the block resembles an MLP applied at each spatial position:

$$
u_{u,v}
=
W_1 z_{u,v} + b_1,
$$

$$
v_{u,v}
=
\operatorname{GELU}(u_{u,v}),
$$

$$
o_{u,v}
=
W_2 v_{u,v} + b_2.
$$

This is close to a Transformer feed-forward network:

$$
\operatorname{FFN}(x)
=
W_2 \sigma(W_1x+b_1)+b_2.
$$

The difference is the input layout: ConvNeXt keeps a 2D feature map and uses convolution for spatial mixing.

## Macro Design

ConvNeXt updates the macro design of ResNet-like models to better match modern vision backbones.

| Axis | Older ResNet Habit | ConvNeXt Direction |
| --- | --- | --- |
| stem | aggressive early convolution/pooling | patchify-like stem |
| stage ratio | ResNet-style stage depths | adjusted stage compute distribution |
| downsampling | ResNet transition blocks | separated downsampling layers |
| block width/depth | classic CNN scaling habits | Transformer-era scaling comparison |
| dense prediction | CNN feature pyramid compatibility | still strong for detection/segmentation |

The important reading point is that architecture comparisons are not only block comparisons. The training recipe and macro allocation of compute can change the conclusion.

## Normalization and Activation Placement

Classic CNNs often use BatchNorm and ReLU repeatedly:

$$
\operatorname{Conv}
\rightarrow
\operatorname{BN}
\rightarrow
\operatorname{ReLU}.
$$

ConvNeXt moves toward a Transformer-like block style:

$$
\operatorname{DWConv}
\rightarrow
\operatorname{LayerNorm}
\rightarrow
\operatorname{Linear}
\rightarrow
\operatorname{GELU}
\rightarrow
\operatorname{Linear}.
$$

This matters because normalization is part of the architecture contract. A "ConvNet vs Transformer" comparison can be unfair if one side uses a more modern optimization recipe and block layout.

## Relation to Swin Transformer

ConvNeXt should be read against Swin rather than against only old ResNet.

| Axis | Swin Transformer | ConvNeXt |
| --- | --- | --- |
| spatial mixing | shifted window attention | large-kernel depthwise convolution |
| hierarchy | patch merging stages | convolutional stages |
| locality | local windows | local convolution kernels |
| channel mixing | MLP | pointwise linear/conv layers |
| dense prediction | strong backbone | strong backbone |
| adaptive pairwise weights | yes | no |

ConvNeXt's claim is strong because it tests whether a modernized ConvNet can match the practical vision-backbone advantages often credited to Transformers.

## Relation to ResNet

ResNet introduced deep residual learning:

$$
y=x+F(x).
$$

ConvNeXt keeps the residual paradigm but changes the branch design:

$$
F_{\mathrm{ResNet}}
\rightarrow
F_{\mathrm{modern\ ConvNet}}.
$$

This makes ConvNeXt a later chapter in the ResNet family:

$$
\text{ResNet}
\rightarrow
\text{modern training}
\rightarrow
\text{large-kernel depthwise block}
\rightarrow
\text{ConvNeXt}.
$$

The paper is useful because it does not treat "CNN" as a frozen 2015 design.

## Evidence to Read

ConvNeXt's evidence should be read along three axes:

| Evidence | What It Tests |
| --- | --- |
| ImageNet classification | whether modernized ConvNet scales as a classifier |
| COCO detection | whether it works as a general dense-prediction backbone |
| ADE20K segmentation | whether spatial hierarchy and feature maps transfer |
| ablation path from ResNet | which design changes account for the gain |
| comparison to Swin | whether attention is necessary for the observed frontier |

The ablation path is especially important. Without it, ConvNeXt would be just another high-performing backbone. With it, the paper becomes a structured architecture comparison.

## What to Watch in the Ablations

Read each step as a change in one part of the architecture/training contract:

$$
\Delta \text{performance}
\not\equiv
\Delta \text{architecture only}.
$$

| Change Type | Possible Confound |
| --- | --- |
| training recipe | longer schedule, augmentation, regularization |
| macro design | stage ratio and compute allocation |
| block design | kernel size, depthwise conv, expansion ratio |
| normalization | BatchNorm vs LayerNorm behavior |
| activation | ReLU vs GELU and activation count |
| scaling | parameter/FLOP regime |

The paper is valuable because it exposes these confounds explicitly.

## Why This Matters for AI Architecture Reading

ConvNeXt teaches a general lesson:

$$
\text{architecture family comparison}
\neq
\text{single block comparison}.
$$

When a new architecture claims superiority, check:

- training recipe;
- data scale;
- augmentation and regularization;
- resolution and token count;
- macro depth/width distribution;
- kernel or attention implementation;
- hardware and throughput.

This applies beyond vision. The same issue appears when comparing [[concepts/architectures/transformer|Transformers]], [[concepts/architectures/state-space-model|state-space models]], [[concepts/architectures/gnn|GNNs]], and [[concepts/architectures/cnn|CNNs]].

## Failure Modes and Caveats

- ConvNeXt does not remove the value of attention for long-range, content-dependent interaction.
- The paper's conclusions are strongest for vision backbone settings, not arbitrary sequence modeling.
- A modernized ConvNet can be competitive, but deployment speed still depends on kernels, memory format, and hardware.
- The result depends on a modern training recipe; old CNN training baselines are not enough.
- Large-kernel depthwise convolution is local and shared, so it has a different bias from attention even when performance is close.

## Reading Checks

| Question | Why It Matters |
| --- | --- |
| Is the comparison against a modernized CNN baseline? | avoids strawman CNN comparisons |
| Are training recipes matched? | separates architecture from optimization |
| Is the task classification or dense prediction? | backbone quality may transfer differently |
| Is the gain from block design, stage design, or scaling? | identifies the actual contribution |
| Does the claim require adaptive global interaction? | ConvNeXt remains local and convolutional |

## Links

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/swin-transformer|Swin Transformer]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/efficientnet|EfficientNet]]

## One-Line Memory

ConvNeXt is the paper that asks whether ConvNets were obsolete or merely outdated, then modernizes ResNet into a Transformer-era convolutional backbone.
