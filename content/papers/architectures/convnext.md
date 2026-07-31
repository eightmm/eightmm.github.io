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

## Modernization Path

The paper is easiest to reproduce as a sequence of controlled changes rather than as a single final block. The exact ordering matters because it prevents the reader from attributing every gain to the last change.

| Step | Change | Architectural question |
| --- | --- | --- |
| 1 | start from a strong ResNet-style baseline and modern recipe | is the baseline itself competitive? |
| 2 | replace the early stem with a patchify-like convolution | does a less aggressive stem improve the token or feature hierarchy? |
| 3 | move toward a stage ratio used by hierarchical Transformers | where should compute be spent across resolutions? |
| 4 | separate downsampling from the residual block | should resolution changes have their own normalization and projection contract? |
| 5 | use depthwise convolution with a larger kernel | can cheap local mixing provide a useful receptive field? |
| 6 | change normalization and activation placement | are block conventions, rather than attention, responsible for part of the gain? |
| 7 | reduce unnecessary nonlinearities and use an inverted bottleneck | how should channel expansion be allocated? |
| 8 | add Layer Scale and stochastic depth in the final recipe | does optimization and regularization complete the modern backbone? |

The table is a reading device. It does not mean that every step is independent in the original experiments. When reproducing the paper, keep the experiment identifier, baseline, and changed variables explicit.

## Full Block Contract

Let the input to one block be

$$
x\in\mathbb{R}^{H\times W\times C}.
$$

In a channel-last implementation, the block can be written as:

$$
h = \operatorname{DWConv}_{7\times7}(x),
$$

$$
q = \operatorname{LN}(h),
$$

$$
u = W_{\mathrm{up}}q+b_{\mathrm{up}},
\qquad
u\in\mathbb{R}^{H\times W\times 4C},
$$

$$
v = \operatorname{GELU}(u),
$$

$$
z = W_{\mathrm{down}}v+b_{\mathrm{down}},
\qquad
z\in\mathbb{R}^{H\times W\times C},
$$

and, with Layer Scale and stochastic depth,

$$
y = x + \operatorname{DropPath}(\gamma\odot z).
$$

Here $\gamma\in\mathbb{R}^{C}$ is a learned per-channel scale initialized to a small value in the final design. The exact initialization and drop-path schedule belong to the training contract; omitting them while claiming to reproduce the final model changes more than the block diagram suggests.

The block has three distinct responsibilities:

1. `DWConv` mixes spatial neighbors without mixing channels.
2. `LN` changes the feature statistics presented to the channel MLP.
3. The two pointwise projections mix channels at each spatial location.

This decomposition is useful when transferring the block to another modality. A protein residue grid, voxel field, or molecular surface may support local spatial mixing, but the meaning of the neighborhood and the boundary condition must be redefined rather than copied blindly.

## Tensor Shapes Through a Stage

For a stage with spatial size $H_s\times W_s$ and width $C_s$, the residual blocks preserve the shape:

$$
\mathbb{R}^{H_s\times W_s\times C_s}
\longrightarrow
\mathbb{R}^{H_s\times W_s\times C_s}.
$$

The transition between stages changes both resolution and width:

$$
\mathbb{R}^{H_s\times W_s\times C_s}
\xrightarrow{\text{downsampling}}
\mathbb{R}^{H_s/2\times W_s/2\times 2C_s}.
$$

The first stem similarly maps an image $I\in\mathbb{R}^{H\times W\times 3}$ to a lower-resolution feature map. A patchify-like stem can be represented as a convolution with kernel and stride $4$:

$$
X_0=\operatorname{Conv}_{4\times4,\,s=4}(I),
\qquad
X_0\in\mathbb{R}^{H/4\times W/4\times C_1}.
$$

This is not identical to a ViT patch embedding. It produces a spatial feature map and continues with convolutional operations, rather than flattening patches into a sequence for global self-attention.

For the commonly discussed tiny/small/base/large family, the stage widths and depths are scaled versions of the same macro-contract. A paper note should record the exact variant used in an experiment instead of referring to “ConvNeXt” as though it were one fixed parameter count.

| Component | Shape invariant or change | Why it matters |
| --- | --- | --- |
| residual block | $H,W,C\to H,W,C$ | enables identity path and repeated feature refinement |
| downsampling layer | $H,W,C\to H/2,W/2,2C$ | allocates compute to deeper semantic stages |
| depthwise kernel | channel count preserved | spatial mixing cost scales roughly with $k^2C$ |
| expansion MLP | $C\to 4C\to C$ | channel mixing dominates block parameters |
| classifier head | final feature map to pooled representation | task-specific, not part of the reusable backbone |

## Parameter and Compute Accounting

For a depthwise convolution with kernel size $k$ and $C$ channels, the parameter count is approximately

$$
P_{\mathrm{dw}}=k^2C.
$$

For the two pointwise projections with expansion ratio $r$,

$$
P_{\mathrm{mlp}}
\approx
rC^2+rC^2
=2rC^2,
$$

ignoring biases. For a feature map with $N=HW$ locations, the multiply-add cost has the same leading structure:

$$
\operatorname{FLOPs}_{\mathrm{block}}
\propto
Nk^2C + 2NrC^2.
$$

This explains two practical facts:

- a larger depthwise kernel increases local mixing cost linearly in $C$ and quadratically in $k$;
- the inverted-bottleneck MLP can dominate parameter and arithmetic cost when $C$ is large.

The absence of a quadratic token-pair term does not automatically make the model faster than attention. Runtime also depends on memory traffic, tensor layout, kernel fusion, accelerator support, batch size, and input resolution.

## Why the Inverted Bottleneck Helps

The block widens the channel dimension during the pointwise transformation:

$$
C\rightarrow rC\rightarrow C,
\qquad r\approx4.
$$

The spatial operator itself remains depthwise, so the expanded channels do not require a full $k\times k$ convolution across all channel pairs. This separates spatial and channel costs:

$$
\text{spatial mixing}: O(Nk^2C),
\qquad
\text{channel mixing}: O(NrC^2).
$$

The expansion is therefore not merely a larger hidden layer. It changes where representation capacity is placed. A narrow channel interface preserves the residual shape while allowing a richer intermediate channel basis.

When implementing a variant, report whether the expansion is before or after the depthwise convolution. Moving it changes activation memory, parameter cost, and the effective block family.

## Downsampling Is a Separate Contract

In many older CNNs, downsampling is folded into a strided convolution inside a residual block. ConvNeXt treats resolution change as an explicit layer between stages:

$$
X_{s+1}
=
\operatorname{Conv}_{2\times2,\,s=2}
\left(\operatorname{LN}(X_s)\right).
$$

The exact ordering is important because a normalization layer sees a different distribution before and after the spatial projection. It also makes feature-pyramid extraction easier to describe:

$$
\{X_1,X_2,X_3,X_4\}
$$

can be exposed at well-defined resolutions for detection and segmentation heads.

For dense prediction, the backbone interface is not just the final classification vector. The relevant contract is:

| Output | Typical use |
| --- | --- |
| early high-resolution feature | edges, local texture, fine localization |
| middle-stage feature | object parts and medium-scale structure |
| late low-resolution feature | semantic context and category evidence |
| ordered multi-scale pyramid | FPN-style detection or segmentation neck |

This is one reason a hierarchical ConvNet remains attractive even when a vanilla ViT has a strong classification score.

## LayerNorm, Layout, and Numerical Details

LayerNorm over channels for a feature vector $x_{u,v}\in\mathbb{R}^{C}$ is:

$$
\operatorname{LN}(x_{u,v})
=
\gamma\odot
\frac{x_{u,v}-\mu_{u,v}}
{\sqrt{\sigma^2_{u,v}+\epsilon}}
+\beta,
$$

where

$$
\mu_{u,v}=\frac1C\sum_{c=1}^{C}x_{u,v,c},
\qquad
\sigma^2_{u,v}=\frac1C\sum_{c=1}^{C}(x_{u,v,c}-\mu_{u,v})^2.
$$

The mathematical operation is independent of whether the tensor is stored as NCHW or NHWC, but the implementation is not. A common implementation permutes NCHW to channel-last for LayerNorm and then permutes back. That can introduce layout conversions and affect measured throughput.

Checklist for a faithful block implementation:

- confirm the normalization axes;
- confirm whether the depthwise convolution uses groups equal to $C$;
- confirm the expansion ratio and linear-layer bias;
- confirm GELU variant and numerical precision;
- confirm Layer Scale placement relative to DropPath;
- confirm stage-transition normalization and stride;
- measure both training memory and inference latency.

## Training Recipe Is Part of the Comparison

ConvNeXt's argument is specifically about a modernized ConvNet, so the training setup cannot be treated as irrelevant metadata. At minimum, record:

| Training variable | Why it can change the conclusion |
| --- | --- |
| epoch budget | longer optimization benefits may differ by architecture |
| optimizer and weight decay | normalization and residual parameterization interact with optimization |
| learning-rate schedule | convergence speed and final accuracy are coupled |
| data augmentation | stronger regularization may narrow the architecture gap |
| label smoothing and mixup/cutmix | changes the effective supervision signal |
| stochastic depth | depth-dependent regularization changes the block behavior |
| input resolution and crop policy | changes token count, receptive-field coverage, and cost |
| pretraining data | can dominate the apparent backbone advantage |

The fair question is therefore not “does ConvNeXt beat an old ResNet?” but:

$$
\text{same data, recipe, budget, resolution, and evaluation}
\quad\Longrightarrow\quad
\text{what remains attributable to the backbone?}
$$

## Ablation Reading Matrix

Use the following matrix when reading the original ablation sequence or implementing a local reproduction. The right-hand column states the strongest claim that the experiment can support.

| Ablation family | Keep fixed | Vary | Supported conclusion |
| --- | --- | --- | --- |
| recipe modernization | architecture | optimizer, schedule, augmentation | old recipe understated the baseline |
| stem | recipe and stage budget | stem kernel/stride | early tokenization affects the hierarchy |
| downsampling | stage widths and depths | transition placement and normalization | resolution changes deserve an explicit interface |
| kernel size | all other block choices | depthwise kernel size | local receptive-field choice matters |
| activation count | width and depth | ReLU/GELU and placement | nonlinearity placement is part of block design |
| bottleneck ratio | stage compute budget | expansion ratio | channel capacity and cost trade off |
| normalization | recipe | BatchNorm/LayerNorm and placement | statistics and optimization interact with architecture |
| macro stage ratio | total budget | depth allocation by stage | global compute allocation matters |
| regularization | architecture | Layer Scale, DropPath, augmentation | final quality is not a pure block effect |

Do not read a one-variable comparison as causal if the change also alters parameter count, FLOPs, activation memory, or training stability. Add a budget-matched control when possible.

## Classification Versus Dense Prediction

ConvNeXt's architecture claim is broader than ImageNet classification, but the evidence should be separated by interface:

$$
\text{image}
\rightarrow
\text{hierarchical backbone}
\rightarrow
\begin{cases}
\text{global pooling + classifier}\\
\text{feature pyramid + detection head}\\
\text{multi-scale features + segmentation head}
\end{cases}
$$

Classification mainly tests the quality of the final representation. Detection and segmentation additionally test spatial resolution, multi-scale alignment, and neck compatibility. A single top-1 score cannot establish the latter claims.

When comparing to Swin or ViT, record whether the downstream framework, neck, pretraining checkpoint, and fine-tuning schedule are matched. Otherwise, “backbone comparison” may actually compare an entire ecosystem.

## Architecture Versus Inductive Bias

ConvNeXt does not erase the difference between convolution and attention. It narrows a practical performance gap under a particular vision regime.

| Property | ConvNeXt | Attention-based backbone |
| --- | --- | --- |
| neighborhood | fixed local offsets | content-dependent selected tokens |
| translation sharing | explicit convolutional weight sharing | depends on tokenization and positional design |
| interaction range per block | kernel-limited | window- or sequence-limited by attention pattern |
| locality prior | strong | weaker or explicitly reintroduced |
| hardware primitive | convolution/depthwise convolution | matrix multiplication plus softmax or sparse kernels |
| geometry transfer | natural for regular grids | requires token/position design |

The right conclusion is conditional:

$$
\text{ConvNeXt competitive on vision benchmarks}
\not\Rightarrow
\text{convolution is universally superior}.
$$

The inductive bias still matters when the input is irregular, the required interaction is global, or the task benefits from content-dependent routing.

## Transfer to Scientific and Biological Data

ConvNeXt is an architecture paper, so any scientific transfer should begin from the data geometry rather than from the model name.

| Input representation | Possible ConvNeXt-like use | Main caveat |
| --- | --- | --- |
| 2D microscopy image | image encoder or dense segmentation backbone | biological scale and acquisition shift |
| voxelized molecular field | local 3D convolutional extension | resolution and rotational symmetry |
| protein contact or distance map | 2D relational map encoder | map is not the full 3D object |
| molecular surface raster | local surface-feature encoder | discretization and orientation choices |
| sequence or graph | generally not a direct ConvNeXt input | regular-grid locality may be artificial |

For protein or molecular coordinates, a regular-grid ConvNet can violate the desired transformation behavior. If rotations or translations should preserve the prediction, use an explicitly invariant or equivariant architecture such as [[papers/architectures/egnn|EGNN]] or a suitable geometric model. ConvNeXt can still serve as a baseline for a rasterized representation, but it should not be presented as a geometry-preserving model by default.

## Reproduction Specification

Before running a reproduction, write the specification below in a machine-readable experiment record:

| Field | Required value |
| --- | --- |
| variant | Tiny, Small, Base, Large, or custom stage configuration |
| input | resolution, channels, crop policy |
| stem | kernel, stride, output width |
| stages | depth and width of each stage |
| block | depthwise kernel, expansion ratio, activation, normalization |
| transition | downsampling kernel, stride, normalization placement |
| residual | Layer Scale initialization and DropPath schedule |
| recipe | optimizer, schedule, epochs, augmentation, weight decay |
| precision | fp32, bf16, fp16, or mixed policy |
| evaluation | dataset split, metric, crop/test protocol |
| system | accelerator, batch size, layout, compiler/kernel settings |

Minimum acceptance checks:

1. A single block preserves $H\times W\times C$.
2. A stage transition halves spatial resolution and applies the documented width change.
3. Depthwise convolution has `groups == channels`.
4. The residual branch and identity branch have compatible layouts.
5. Parameter and FLOP counts agree with the declared variant within the tool's counting convention.
6. Classification and dense-prediction feature maps are taken from the documented stages.
7. Accuracy and throughput are reported with the same evaluation and system conditions as the baseline.

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
