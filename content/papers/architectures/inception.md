---
title: Going Deeper with Convolutions
aliases:
  - papers/inception
  - papers/going-deeper-with-convolutions
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Going Deeper with Convolutions

> The paper introduced the Inception module as a multi-branch CNN block for increasing depth and width under a compute budget.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Going Deeper with Convolutions |
| Authors | Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich |
| Year | 2015 |
| Venue | CVPR 2015 |
| arXiv | [1409.4842](https://arxiv.org/abs/1409.4842) |
| CVF | [CVPR 2015 paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) |
| Status | verified |

## Question

After [[papers/architectures/alexnet|AlexNet]] and around the same era as [[papers/architectures/vgg|VGG]], CNNs were getting deeper and wider. But naive scaling increases compute and parameters quickly:

$$
\text{more channels}
+
\text{larger kernels}
+
\text{more layers}
\Rightarrow
\text{higher cost and overfitting risk}.
$$

Inception asks:

$$
\text{Can a CNN block use multiple receptive-field sizes while keeping computation practical?}
$$

The paper's answer is a multi-branch module with $1\times1$ projections that control channel count before expensive convolutions.

## Main Claim

An Inception module can improve large-scale vision performance by combining parallel convolution and pooling branches under a compute-aware design.

The abstract module is:

$$
y
=
\operatorname{Concat}
\left[
f_{1\times1}(x),
f_{3\times3}(x),
f_{5\times5}(x),
p(x)
\right].
$$

In the practical module, expensive branches use $1\times1$ reductions first:

$$
f_{3\times3}(x)
=
\operatorname{Conv}_{3\times3}
(
\operatorname{Conv}_{1\times1}(x)
)
$$

$$
f_{5\times5}(x)
=
\operatorname{Conv}_{5\times5}
(
\operatorname{Conv}_{1\times1}(x)
).
$$

The durable architecture claim is:

$$
\text{parallel receptive fields}
+
\text{channel bottlenecks}
+
\text{concatenation}
\Rightarrow
\text{wider/deeper CNNs under a practical compute budget}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | image feature map $X\in\mathbb{R}^{H\times W\times C}$ |
| Output | concatenated feature map from multiple branches |
| Core block | parallel $1\times1$, $3\times3$, $5\times5$, and pooling branches |
| Projection | $1\times1$ convolutions reduce channel count before expensive kernels |
| Merge | channel concatenation |
| Downstream task | ImageNet classification and detection system |
| Design target | increase width/depth without exploding compute |
| Main bias | multi-scale local visual features |

The block keeps the spatial grid while changing channel composition:

$$
X
\to
[B_1(X),B_2(X),B_3(X),B_4(X)]
\to
\operatorname{Concat}_{\text{channel}}.
$$

The output channel dimension is:

$$
C_{\text{out}}
=
C_{1\times1}
+
C_{3\times3}
+
C_{5\times5}
+
C_{\text{pool-proj}}.
$$

## Why Multi-Branch Convolutions?

Different visual patterns live at different spatial scales. A $1\times1$ branch captures channel mixing at each location. A $3\times3$ branch captures local spatial structure. A $5\times5$ branch captures larger local context. A pooling branch provides local invariance and summary.

| Branch | Reads | Typical Role |
| --- | --- | --- |
| $1\times1$ conv | same spatial position, all channels | channel mixing and cheap features |
| $3\times3$ conv | local neighborhood | medium local pattern |
| $5\times5$ conv | larger local neighborhood | larger texture/part pattern |
| pooling + projection | local region | invariance and summary |

Concatenation lets the next layer decide how to combine these features.

## 1x1 Convolution As Projection

A $1\times1$ convolution is a learned linear projection applied independently at each spatial position:

$$
Y_{u,v,k}
=
\sum_{c=1}^{C_{\text{in}}}
W_{c,k}
X_{u,v,c}
+
b_k.
$$

It does not mix neighboring spatial positions. It mixes channels.

This makes it useful as a bottleneck before expensive convolutions. If a $5\times5$ convolution maps $C_{\text{in}}$ channels to $C_{\text{out}}$ channels, the parameter count is:

$$
25C_{\text{in}}C_{\text{out}}.
$$

If a $1\times1$ projection first reduces channels to $C_r$, the count becomes:

$$
C_{\text{in}}C_r
+
25C_rC_{\text{out}}.
$$

When $C_r\ll C_{\text{in}}$, this is much cheaper.

## Compute-Aware Width And Depth

Inception is not just "add many branches." Without bottlenecks, the branches would be expensive. The architecture is compute-aware:

$$
\text{wide multi-branch block}
\quad\text{is feasible only if}\quad
\text{branch channels are controlled}.
$$

This is why the paper belongs beside [[concepts/architectures/computational-complexity|computational complexity]] notes. The architectural idea includes resource allocation:

- which receptive-field sizes get capacity;
- how many channels each branch receives;
- where to reduce dimensions;
- when to downsample;
- how to keep classifier parameters manageable.

## Block View

| Component | Role | Architecture Implication |
| --- | --- | --- |
| $1\times1$ branch | cheap channel mixing | keeps local position, changes feature basis |
| $3\times3$ branch | local spatial features | medium receptive field |
| $5\times5$ branch | larger local features | higher cost, needs reduction |
| pooling branch | local summary/invariance | complements convolution branches |
| $1\times1$ reduction | bottleneck before expensive conv | compute control |
| concatenation | merge branch outputs | widens feature representation |
| auxiliary classifiers | extra training signal | historical deep-training aid |
| global average pooling | reduce classifier parameters | less dense-head overfitting |

The block is modular but manually designed. Later architectures searched, scaled, factorized, or residualized these ideas.

## Auxiliary Classifiers

The paper uses auxiliary classifiers during training to inject additional gradient signal into intermediate layers. Conceptually:

$$
\mathcal{L}
=
\mathcal{L}_{\text{main}}
+
\lambda_1\mathcal{L}_{\text{aux},1}
+
\lambda_2\mathcal{L}_{\text{aux},2}.
$$

This helps train a deeper model under the optimization recipes of the time. Later residual connections and normalization reduced the need for this kind of auxiliary supervision in many CNN backbones.

When reading results, treat auxiliary classifiers as part of the training recipe, not just architecture topology.

## Global Average Pooling

Instead of ending with a very large fully connected classifier, Inception uses global average pooling near the end:

$$
h_c
=
\frac{1}{HW}
\sum_{u=1}^{H}
\sum_{v=1}^{W}
X_{u,v,c}.
$$

This reduces spatial maps to a channel vector. Compared with a large flattened fully connected head, it lowers parameter count and encourages channels to act like class-relevant feature detectors.

This is one reason Inception is more parameter-efficient than older CNNs with heavy dense heads.

## Evidence Reading

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Inception improves ImageNet classification | ILSVRC performance | multi-branch compute-aware CNN is strong | architecture, training, and ensemble/system details interact |
| Inception helps detection | detection challenge results | learned features transfer to detection pipeline | detection system contains more than backbone |
| $1\times1$ reductions make wide modules feasible | module design and parameter budget | channel bottlenecks control compute | exact branch widths are manually tuned |
| Auxiliary classifiers help training | training recipe | deep CNN optimization benefited from extra loss | later ResNet/BatchNorm recipes changed this need |

Read the paper as a system milestone plus a reusable module idea. Do not reduce it to "parallel convolutions" only.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | ImageNet classification and detection |
| Input/output unit | image to class label or detection pipeline feature |
| Architecture family | CNN |
| Main block | Inception module |
| Main metric | challenge classification/detection performance |
| Main design variable | multi-branch width/depth under compute budget |
| Not directly tested | modern self-supervised pretraining, ViT-scale pretraining, 3D or graph data |

## Relation To VGG

[[papers/architectures/vgg|VGG]] makes CNN design simple and sequential:

$$
3\times3
\to
3\times3
\to
3\times3.
$$

Inception makes CNN design multi-branch and compute-aware:

$$
\operatorname{Concat}
[
1\times1,
3\times3,
5\times5,
\operatorname{Pool}
].
$$

| Dimension | VGG | Inception |
| --- | --- | --- |
| Design style | uniform sequential stack | multi-branch module |
| Main question | does depth help with small filters? | can width/depth be allocated efficiently? |
| Simplicity | high | lower |
| Parameter efficiency | poor by modern standards | stronger |
| Receptive fields | grow through depth | multiple branch scales per module |

VGG is cleaner as a baseline. Inception is more resource-aware.

## Relation To ResNet

[[papers/architectures/deep-residual-learning|ResNet]] later changed the depth-scaling problem with identity paths:

$$
y = x + F(x).
$$

Inception asks how to design the function $F$ as a multi-branch module. ResNet asks how to make very deep stacks of functions trainable.

Later architectures combine both ideas: residual paths around multi-branch or bottleneck modules.

## Relation To Efficient CNN Design

Inception anticipates several later CNN design themes:

- bottlenecks before expensive operations;
- factorized computation;
- lower-parameter classifier heads;
- balancing depth, width, and resolution;
- architecture as compute allocation.

[[papers/architectures/efficientnet|EfficientNet]] later formalizes scaling of depth, width, and resolution. Inception is an earlier manually designed compute-aware architecture.

## Relation To Vision Transformers

Inception relies on local convolutional branches. [[papers/architectures/vision-transformer|Vision Transformer]] replaces this with token mixing through self-attention:

| Dimension | Inception | Vision Transformer |
| --- | --- | --- |
| Feature unit | spatial feature map | patch tokens |
| Mixing | local branch convolutions | global self-attention |
| Multi-scale behavior | manually designed branches | learned through layers/heads and later hierarchical designs |
| Compute issue | branch channel allocation | attention token count |
| Bias | strong local image prior | weaker local prior unless modified |

Inception is useful for seeing how much work CNNs did to encode local multi-scale structure before attention-based vision matured.

## Implementation Notes

When reading or reproducing Inception-style claims, check:

| Detail | Why It Matters |
| --- | --- |
| branch channel counts | determines compute and output width |
| reduction dimensions | controls bottleneck strength |
| pooling branch projection | affects output balance |
| auxiliary classifiers | training-only component changes optimization |
| global average pooling | reduces dense head parameters |
| ensemble or single model | challenge results often involve system-level choices |
| input preprocessing | affects ImageNet numbers |
| detection pipeline | backbone gain and detector gain can mix |

The exact branch widths are not a universal law. They are design choices under a compute budget.

## Common Misreadings

### "Inception just runs different convolutions in parallel"

Parallel branches are visible, but the key is compute-aware branch design with $1\times1$ channel reductions.

### "1x1 convolution is spatial convolution"

It does not mix spatial neighbors. It mixes channels at each spatial location.

### "Auxiliary classifiers are part of inference"

They are mainly training aids. The main inference model uses the primary classifier.

### "Inception is simpler than VGG"

Inception is more efficient but less simple. VGG is uniform; Inception has branch-width design choices.

## What To Check In Later CNN Papers

- Does the method use multi-branch modules, bottlenecks, or residual paths?
- Are parameter count and FLOPs reported?
- Is the gain from architecture, scaling, training recipe, or ensemble?
- Are branch widths manually tuned?
- Does the model preserve spatial resolution for dense prediction?
- Are auxiliary losses used only during training?
- Is the result single-model or multi-model/system-level?

## Why It Still Matters

Inception is the canonical reference for compute-aware multi-branch CNN modules. It makes an architectural point that remains relevant:

$$
\text{architecture design}
=
\text{information path design}
+
\text{compute allocation}.
$$

For this wiki, it links the classic vision backbone path:

- [[papers/architectures/alexnet|AlexNet]]: large CNNs work.
- [[papers/architectures/vgg|VGG]]: simple depth with small filters works.
- Inception: multi-scale branches can be efficient.
- [[papers/architectures/deep-residual-learning|ResNet]]: identity paths make very deep networks trainable.

## Limitations

- Many branch-width choices are manually designed.
- Architecture evidence is mixed with training, challenge system, and ensemble details.
- Later residual and normalization methods changed the default CNN recipe.
- The module is specialized for grid-like image feature maps.
- It is less clean as a baseline than VGG and less scalable as a default than ResNet-style families.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/modalities/image|Image]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/vgg|VGG]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/efficientnet|EfficientNet]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/index|Architecture papers]]
