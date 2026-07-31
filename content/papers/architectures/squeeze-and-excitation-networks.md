---
title: Squeeze-and-Excitation Networks
aliases:
  - papers/senet
  - papers/se-net
  - papers/squeeze-excitation
tags:
  - papers
  - architectures
  - cnn
  - vision
  - attention
---

# Squeeze-and-Excitation Networks

> The paper made channel-wise feature recalibration a reusable CNN block.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Squeeze-and-Excitation Networks |
| Authors | Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu |
| Year | 2018 |
| Venue | CVPR 2018 |
| arXiv | [1709.01507](https://arxiv.org/abs/1709.01507) |
| CVF | [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) |
| Status | full paper note |

## Question

Convolutions mix spatial and channel information locally, but standard CNN blocks usually treat output channels as a fixed set of feature maps. SENet asks:

$$
\text{Can a CNN adaptively reweight channels based on the current input?}
$$

The answer is a small squeeze-and-excitation block:

$$
\text{global summary}
\rightarrow
\text{channel gate}
\rightarrow
\text{feature recalibration}.
$$

## Main Claim

Squeeze-and-Excitation blocks explicitly model channel interdependencies and improve CNN representations with small additional cost.

The durable claim is:

$$
\text{spatial convolution}
+
\text{input-dependent channel gate}
\Rightarrow
\text{stronger CNN feature hierarchy}.
$$

This is a reusable architecture-block paper.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | feature map $U\in\mathbb{R}^{H\times W\times C}$ |
| Output | recalibrated feature map $\tilde{U}\in\mathbb{R}^{H\times W\times C}$ |
| Squeeze | global average pooling over spatial positions |
| Excitation | small bottleneck MLP producing channel weights |
| Recalibration | channel-wise multiplication |
| Insert point | can be added to many CNN blocks |
| Main bias | input-dependent channel importance |

## Squeeze

Given feature map:

$$
U\in\mathbb{R}^{H\times W\times C},
$$

the squeeze operation computes a channel descriptor:

$$
z_c
=
\frac{1}{H W}
\sum_{u=1}^{H}\sum_{v=1}^{W}
U_{u,v,c}.
$$

This produces:

$$
z\in\mathbb{R}^{C}.
$$

The descriptor is global over space and channel-specific.

## Excitation

The excitation network maps the channel descriptor to a gate:

$$
s
=
\sigma
\left(
W_2 \delta(W_1 z)
\right),
$$

where:

| Symbol | Meaning |
| --- | --- |
| $W_1$ | reduce channels from $C$ to $C/r$ |
| $\delta$ | nonlinearity such as ReLU |
| $W_2$ | expand channels from $C/r$ to $C$ |
| $\sigma$ | sigmoid gate |
| $r$ | reduction ratio |

The output is:

$$
s\in(0,1)^C.
$$

## Recalibration

Each channel is rescaled:

$$
\tilde{U}_{u,v,c}
=
s_c U_{u,v,c}.
$$

This lets the block emphasize or suppress channels depending on the input.

| Step | Operation | Effect |
| --- | --- | --- |
| squeeze | spatial global pooling | summarize channel response |
| excitation | bottleneck MLP | model channel dependency |
| scale | channel-wise multiplication | recalibrate features |

## Full Block Contract

Let $U$ be the output of a spatial feature transform, such as a convolutional or residual branch:

$$
U=F(X)\in\mathbb{R}^{B\times H\times W\times C}.
$$

The SE block computes:

$$
z=\operatorname{GAP}(U),
$$

$$
s=\operatorname{sigmoid}(W_2\delta(W_1z)),
$$

$$
\tilde{U}=s\odot U.
$$

The same gate vector $s$ is broadcast over all spatial positions in a sample. It is input-dependent but spatially constant:

$$
\tilde{U}_{b,u,v,c}=s_{b,c}U_{b,u,v,c}.
$$

This defines the central boundary of the mechanism:

| Property | SE behavior |
| --- | --- |
| sample dependence | each sample can receive a different gate |
| spatial dependence | one weight per channel, shared over $H\times W$ |
| channel dependence | gates are jointly predicted from the pooled descriptor |
| output shape | unchanged from input feature map |
| parameter sharing | same excitation MLP used at every spatial location |

SE is therefore a channel-wise modulation layer, not a spatial map generator.

## Reduction Ratio and Parameter Cost

With reduction ratio $r$, the excitation MLP has dimensions:

$$
C
\rightarrow
\frac{C}{r}
\rightarrow
C.
$$

Ignoring biases, its parameter count is approximately:

$$
P_{\text{SE}}
=
C\frac{C}{r}
+
\frac{C}{r}C
=
\frac{2C^2}{r}.
$$

The spatial squeeze costs:

$$
C_{\text{pool}}
\propto
HWC,
$$

and the gate computation is independent of spatial area after pooling. The recalibration itself applies one scalar multiplication per activation:

$$
C_{\text{scale}}
\propto
HWC.
$$

The reduction ratio controls the capacity/overhead tradeoff:

| Smaller $r$ | Larger $r$ |
| --- | --- |
| wider excitation hidden layer | narrower hidden layer |
| richer channel dependency model | lower parameter and compute cost |
| potentially stronger recalibration | potentially underpowered gate |
| larger memory/latency overhead | cheaper insertion into many blocks |

The cost is small relative to a wide convolution in many settings, but it is not zero and can matter at small batch or edge deployment.

## Channel Dependency Modeling

The squeeze descriptor contains one statistic per channel:

$$
z_c=\mathbb{E}_{(u,v)}[U_{u,v,c}].
$$

The excitation layer predicts each gate from all channel descriptors:

$$
s_c
=
\sigma\left(
W_{2,c:}
\delta(W_1z)
\right).
$$

Therefore the gate for channel $c$ can depend on the activation of channel $c'$:

$$
\frac{\partial s_c}{\partial z_{c'}}
\ne 0
$$

when the MLP connects those dimensions. The block is not independent scalar normalization. It is a low-dimensional learned interaction model over channel summaries.

## Where to Insert the SE Block

The SE unit can be inserted after a transform and before the residual merge:

$$
y=x+\operatorname{SE}(F(x)).
$$

Alternatively, it can modulate a block output before a projection or be used in a non-residual CNN stage. The insertion point changes the semantics:

| Placement | Gate controls |
| --- | --- |
| after residual branch, before addition | contribution of the learned residual transform |
| after residual addition | the complete block state, including identity features |
| inside bottleneck | intermediate channels only |
| after depthwise convolution | spatially filtered channel responses |
| before classifier | late semantic channels |

The original paper demonstrates the block as a modular addition to different CNN families. A reproduction should state the placement rather than only saying “SE was added.”

## SE-ResNet Composition

For a residual block:

$$
F(x)=W_3*\,\delta(W_2*\,\delta(W_1*x)),
$$

an SE residual block can be expressed as:

$$
y=x+\operatorname{SE}(F(x)).
$$

Expanding the definition:

$$
y
=
x
+
s(F(x))\odot F(x).
$$

The identity path remains unscaled in this placement. This gives the gate a residual-branch interpretation:

$$
\text{identity contribution}
+
\text{input-dependent residual contribution}.
$$

If SE is moved after addition, the identity path is also gated, which is a different model.

## Gate Semantics

The sigmoid produces gates in $(0,1)$:

$$
0<s_c<1.
$$

The mechanism can suppress a channel strongly or preserve it near its original magnitude, but it does not amplify it above one without additional rescaling or a different gate parameterization. This is a useful distinction from attention mechanisms whose normalized weights may redistribute mass or from unconstrained feature modulation layers that can amplify arbitrarily.

The gate should not be interpreted as a causal explanation of channel importance automatically. It is an internal control signal learned to optimize the task loss. Diagnostic claims require perturbation or ablation experiments.

## Relation to Normalization

SE and normalization solve different problems:

| Mechanism | Main operation | Input dependence |
| --- | --- | --- |
| BatchNorm | normalize feature statistics and affine-transform channels | batch/training statistics plus learned affine terms |
| LayerNorm | normalize across selected feature dimensions | per sample/token statistics |
| SE | multiply channels using a learned global descriptor | per sample, from pooled activations |

They can coexist. A typical block may normalize and activate a convolution before SE, then pass the recalibrated result to a residual merge. Replacing one with the other changes both the scale and the conditioning behavior.

## SE Versus Spatial Attention and Self-Attention

The phrase “attention” covers different axes:

| Mechanism | Summary | Weight shape or domain |
| --- | --- | --- |
| SE | global channel recalibration | $B\times C$ |
| spatial attention | position-dependent feature gating | $B\times H\times W$ |
| channel-spatial attention | separate or joint channel/position gates | $B\times H\times W\times C$ or factorized |
| self-attention | pairwise token interaction | queries/keys/values over positions or tokens |

SE uses a global spatial summary, then applies a channel gate. It cannot directly model which pixel should attend to which other pixel. The distinction matters when mapping the idea to image patches, molecular atoms, graph nodes, or sequence tokens.

## Evidence and Claim Boundaries

The paper's evidence supports a modular channel-recalibration claim:

| Claim | Evidence | Boundary |
| --- | --- | --- |
| SE improves CNN representations | insertion into multiple strong CNN families | improvement depends on placement and training recipe |
| block is reusable | works with different backbones | not every architecture benefits equally |
| overhead is small | parameter/compute comparisons | edge latency and memory still require measurement |
| channel dependencies matter | channel-gating design and ablations | gate values are not automatically explanations |
| gains transfer across datasets | reported classification experiments | not proof for every modality or task |

The architecture contribution is the explicit module contract. The exact benchmark gain should be attributed jointly to backbone, training setup, and insertion strategy.

## Ablation Matrix

| Ablation | Question | What it isolates |
| --- | --- | --- |
| remove squeeze | is global channel context necessary? | local transform versus pooled descriptor |
| remove excitation MLP | are fixed or independent gates enough? | channel dependency modeling |
| reduction ratio $r$ | how much gate capacity is useful? | overhead/capacity tradeoff |
| sigmoid versus other gate | does bounded gating matter? | modulation parameterization |
| placement before/after residual merge | which state should be recalibrated? | residual branch versus full state |
| global average versus other pooling | which summary statistic is useful? | spatial information loss |
| channel-only versus spatial attention | which axis carries the gain? | channel recalibration versus localization |
| matched compute baseline | is the gain worth the overhead? | architecture benefit under budget |

The fairest baseline keeps the backbone and training recipe fixed, adds only the SE module, and reports both accuracy and overhead.

## Implementation Pitfalls

1. **Wrong pooling axes**: squeeze over spatial dimensions but preserve batch and channel dimensions.
2. **Gate shape mismatch**: reshape $s$ to broadcast over height and width, not over channels accidentally.
3. **Placement drift**: applying SE after residual addition is not equivalent to gating the residual branch.
4. **Reduction rounding**: $C/r$ must be rounded and kept at a valid minimum width.
5. **Activation mismatch**: changing the excitation nonlinearity or gate function changes the block contract.
6. **Silent amplification assumptions**: sigmoid gates suppress or preserve magnitudes; they do not independently amplify above one.
7. **Overinterpreting gates**: high gate values are not causal feature importance without intervention tests.
8. **Ignoring deployment overhead**: small parameter count does not guarantee zero latency or memory cost.

For a minimal test, feed a feature map with one channel selectively increased, verify that the gate vector changes, and check that the output is exactly the broadcast elementwise product. Then compare residual-branch and post-addition placements separately.

## Transfer to Efficient and Scientific Models

SENet's channel gate appears naturally in later efficient CNN blocks, including MBConv-style families. The general composition is:

$$
\text{expand}
\rightarrow
\text{spatial filter}
\rightarrow
\text{channel recalibration}
\rightarrow
\text{project}.
$$

For scientific or biological inputs, channel semantics need additional care:

- channels may represent distinct measured modalities rather than exchangeable learned features;
- global average pooling may remove spatial or structural information needed by the target;
- vector/tensor channels may require equivariant gating rather than independent scalar gates;
- graph nodes and molecular atoms are not spatial grid channels;
- gate values should be validated against perturbation or held-out distribution shifts.

The transferable idea is input-conditioned feature selection. The literal squeeze operation is appropriate only when global pooling preserves the relevant task information.

## Reproduction Checklist

- [ ] record the feature-map shape and squeeze axes;
- [ ] record reduction ratio, hidden width rounding, and excitation activation;
- [ ] record gate function and its output range;
- [ ] record exact insertion point relative to normalization, activation, and residual addition;
- [ ] verify broadcast multiplication over spatial dimensions;
- [ ] report added parameters, operations, memory, and latency;
- [ ] compare with a same-backbone no-SE baseline;
- [ ] sweep reduction ratio and placement;
- [ ] separate channel-gate diagnostics from causal feature-importance claims;
- [ ] test whether global pooling is valid for the target modality.

## Relation to Attention

SE blocks are often described as channel attention. They do not compute token-token attention like [[concepts/architectures/attention|attention]] in Transformers. Instead, they compute a channel gate:

$$
\text{SE: } U \mapsto s(U)\odot U.
$$

The similarity is input-dependent weighting. The difference is the axis:

| Mechanism | Weighted Axis |
| --- | --- |
| SE block | channels |
| spatial attention | spatial positions |
| self-attention | tokens or patches |

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| ImageNet classification | SE blocks improve strong CNN backbones |
| insertion into existing networks | the block is modular rather than a full-only architecture |
| ILSVRC 2017 result | SE-based models were competitive at large scale |

## Limits

- Global average pooling discards spatial arrangement before channel gating.
- The block improves channel recalibration, but does not replace spatial modeling.
- The extra MLP is small but not zero-cost.
- Gains can depend on baseline strength, training recipe, and where the block is inserted.

## Concepts

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/computational-complexity|Computational complexity]]

## Related

- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/resnext|ResNeXt]]
- [[papers/architectures/mobilenetv2|MobileNetV2]]
- [[papers/architectures/efficientnet|EfficientNet]]
