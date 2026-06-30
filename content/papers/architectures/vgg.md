---
title: Very Deep Convolutional Networks for Large-Scale Image Recognition
aliases:
  - papers/vgg
  - papers/very-deep-convolutional-networks
tags:
  - papers
  - architectures
  - cnn
  - vision
---

# Very Deep Convolutional Networks for Large-Scale Image Recognition

> The paper showed that depth with small convolution filters is a powerful and simple design rule for CNN backbones.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Very Deep Convolutional Networks for Large-Scale Image Recognition |
| Authors | Karen Simonyan, Andrew Zisserman |
| Year | 2015 |
| Venue | ICLR 2015 |
| arXiv | [1409.1556](https://arxiv.org/abs/1409.1556) |
| Author page | [Oxford VGG publication page](https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/) |
| Status | verified |

## Question

After [[papers/architectures/alexnet|AlexNet]], CNNs had clearly become the dominant direction for large-scale image recognition, but the design space was still messy: large kernels, different layer widths, local response normalization, pooling choices, and hardware-driven implementation details were mixed together.

VGG asks a cleaner architecture question:

$$
\text{What happens if we make the CNN much deeper while using a simple repeated }3\times3\text{ convolution design?}
$$

The paper is important because it turns CNN design into a depth study with a uniform block pattern.

## Main Claim

CNN accuracy improves when depth is increased using small convolution filters and a simple uniform architecture.

The durable claim is:

$$
\text{small local filters}
+
\text{more nonlinear layers}
+
\text{systematic depth increase}
\Rightarrow
\text{stronger image representation}.
$$

VGG is not mainly a new operation paper. It is a design-principle paper: use repeated small convolutions and scale depth.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | fixed-size RGB image crop |
| Output | ImageNet class probability |
| Backbone | sequential CNN with repeated $3\times3$ convolution blocks |
| Downsampling | max pooling between stages |
| Activation | ReLU after convolution |
| Classifier | fully connected head |
| Design variable | network depth, usually 11 to 19 weight layers |
| Main bias | local convolution, translation sharing, hierarchical spatial features |

The high-level structure is:

$$
X
\to
\operatorname{ConvBlock}_1
\to
\operatorname{Pool}
\to
\operatorname{ConvBlock}_2
\to
\cdots
\to
\operatorname{FC}
\to
\operatorname{softmax}.
$$

Each stage keeps the design simple: stack small convolutions, then downsample.

## Why 3x3 Convolutions Matter

A $3\times3$ convolution is the smallest kernel that can capture interactions across both spatial axes and the center:

$$
Y_{u,v,k}
=
\sum_{\Delta u=-1}^{1}
\sum_{\Delta v=-1}^{1}
\sum_c
W_{\Delta u,\Delta v,c,k}
X_{u+\Delta u,v+\Delta v,c}
+
b_k.
$$

Stacking small filters grows the effective receptive field. Two $3\times3$ layers cover a $5\times5$ receptive field:

$$
3\times3
\rightarrow
3\times3
\quad
\Rightarrow
\quad
5\times5\text{ effective field}.
$$

Three $3\times3$ layers cover a $7\times7$ receptive field:

$$
3\times3
\rightarrow
3\times3
\rightarrow
3\times3
\quad
\Rightarrow
\quad
7\times7\text{ effective field}.
$$

The advantage is not only receptive field size. Stacking layers adds nonlinearities:

$$
\phi(W_3\phi(W_2\phi(W_1x)))
$$

is more expressive than one linear convolution followed by one activation.

## Parameter Comparison

Assume input and output channel count are both $C$. A single $7\times7$ convolution has:

$$
7\cdot7\cdot C^2 = 49C^2
$$

parameters. Three $3\times3$ convolutions have:

$$
3\cdot(3\cdot3\cdot C^2)=27C^2
$$

parameters, ignoring biases and channel changes.

So repeated small kernels can provide:

- similar effective receptive field;
- fewer parameters than a large kernel;
- more nonlinear transformations;
- a cleaner uniform design.

This is the central VGG design logic.

## Depth As The Main Variable

VGG compares networks with increasing numbers of weight layers. The paper's architecture study is easier to read than AlexNet because the block pattern is deliberately uniform.

| Design Choice | VGG Reading |
| --- | --- |
| kernel size | mostly $3\times3$ |
| stride | mostly fixed small-step convolution |
| pooling | max pooling between stages |
| width | channel count increases with depth/stage |
| depth | main experimental variable |
| head | large fully connected classifier |

The result is a family of networks where depth can be isolated more clearly than in architecture recipes with many changing knobs.

## Block View

| Component | Role | Architecture Implication |
| --- | --- | --- |
| $3\times3$ convolution | local feature extraction | simple repeated primitive |
| ReLU | nonlinear depth | each small conv adds an activation boundary |
| max pooling | spatial downsampling | grows effective context and reduces compute |
| channel increase | more capacity at lower resolution | common CNN stage pattern |
| fully connected layers | classification head | parameter-heavy by modern standards |
| scale jitter / augmentation | generalization | part of training recipe, not pure architecture |

VGG is simple enough to be a baseline backbone, but not efficient enough to be a modern default.

## Receptive Field

For stride-1 convolutions with kernel size $k$, a simple receptive field approximation is:

$$
R_L = 1 + L(k-1).
$$

For $k=3$:

$$
R_L = 1 + 2L.
$$

Thus:

| Stacked $3\times3$ Layers | Effective Receptive Field |
| --- | --- |
| 1 | $3\times3$ |
| 2 | $5\times5$ |
| 3 | $7\times7$ |
| 4 | $9\times9$ |

Pooling and stride increase the receptive field faster, but the simple stride-1 stack explains the core design idea.

## Evidence Reading

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Depth improves ImageNet performance | comparison across VGG configurations | deeper CNNs help when design is controlled | compute and parameters also increase |
| Small filters are effective | strong results with repeated $3\times3$ blocks | large early kernels are not necessary | later designs improve efficiency |
| Features transfer | evaluation beyond the main benchmark | learned CNN features are reusable | transfer evaluation predates modern foundation-model practice |
| Uniform design is useful | architecture family comparison | simple repeated blocks are strong baselines | plain depth later hits optimization limits |

Read VGG as a depth and design-simplicity paper. It is not a final answer for efficient CNN design.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | image classification |
| Dataset | ImageNet / ILSVRC |
| Input/output unit | image to class label |
| Architecture family | CNN |
| Main metric | classification error |
| Main variable | depth with small convolutions |
| Not directly tested | modern segmentation, detection, video, multimodal learning, molecular grids |

## Relation To AlexNet

[[papers/architectures/alexnet|AlexNet]] proved that large CNNs trained on ImageNet could dominate older vision systems. VGG asks how to make CNN design more systematic.

| Dimension | AlexNet | VGG |
| --- | --- | --- |
| Historical role | ImageNet breakthrough | depth and small-kernel design rule |
| Kernel design | larger early kernels and mixed choices | uniform $3\times3$ convolution |
| Depth | deep for 2012 | much deeper plain CNNs |
| Main lesson | end-to-end deep CNNs work | depth with small filters works |
| Complexity | system milestone | architecture design study |

The transition is:

$$
\text{CNNs can win}
\to
\text{simple deeper CNNs win better}.
$$

## Relation To Inception

[[papers/architectures/inception|Inception]] takes a different route. Instead of using a single uniform stack, it builds multi-branch modules with different receptive-field sizes.

| Dimension | VGG | Inception |
| --- | --- | --- |
| Design style | sequential and uniform | multi-branch module |
| Main primitive | repeated $3\times3$ conv | $1\times1$, $3\times3$, $5\times5$, pooling branches |
| Efficiency | parameter-heavy | more compute-aware |
| Read for | depth and simplicity | multi-scale and factorized compute |

VGG is easier to reason about. Inception is more efficient and modular.

## Relation To ResNet

VGG-style plain depth eventually runs into optimization problems. [[papers/architectures/deep-residual-learning|ResNet]] changes the depth scaling rule:

$$
y = x + F(x).
$$

VGG says:

$$
\text{more stacked convolution layers can improve representation}.
$$

ResNet says:

$$
\text{very deep stacks need identity paths to optimize reliably}.
$$

This makes VGG the natural predecessor to residual networks. It motivates the question that ResNet answers: if depth helps, how do we train much deeper networks?

## Relation To Vision Transformers

VGG is a strong statement of local grid bias. [[papers/architectures/vision-transformer|Vision Transformer]] weakens that bias by treating image patches as tokens.

| Dimension | VGG | Vision Transformer |
| --- | --- | --- |
| Unit | pixels/features on a grid | image patches as tokens |
| Mixing | local convolution | global self-attention |
| Bias | strong locality and translation sharing | weaker image-specific bias |
| Scaling route | depth over CNN blocks | data/compute over attention blocks |
| Receptive field | grows through depth/pooling | global from early layers |

VGG is still useful because it makes the CNN prior explicit and easy to compare against attention-based vision.

## Transfer Feature Role

VGG features became widely reused as visual backbones. A trained CNN can be split into:

$$
h = \operatorname{Backbone}(X)
$$

and:

$$
\hat{y}=\operatorname{Head}(h).
$$

For transfer, the backbone is reused and the head is replaced or fine-tuned. This matters historically because VGG made CNN backbones feel like reusable representation extractors.

Modern foundation models changed the scale and pretraining objectives, but the backbone/head decomposition remains central.

## Implementation Notes

Important details when reading or reproducing VGG-style claims:

| Detail | Why It Matters |
| --- | --- |
| input crop and scale | changes ImageNet accuracy |
| depth configuration | VGG-11, VGG-13, VGG-16, VGG-19 differ |
| fully connected head | dominates parameter count |
| initialization | plain deep networks can be sensitive |
| augmentation | affects generalization |
| no residuals | limits depth scaling |
| no BatchNorm in original core recipe | later VGG-BN variants change training behavior |
| feature transfer setup | task, head, and fine-tuning protocol matter |

When a paper says "VGG backbone," check whether it means original VGG, VGG with BatchNorm, pretrained VGG features, or a truncated VGG feature extractor.

## Common Misreadings

### "VGG is good because it is efficient"

VGG is simple and historically strong, but it is parameter-heavy and inefficient compared with later CNNs.

### "The only contribution is using 3x3 filters"

The contribution is the systematic combination of small filters, depth, and controlled architecture comparison.

### "Deeper plain CNNs always improve"

Depth helps up to a point, but optimization and degradation problems appear without residual connections or better normalization.

### "VGG is obsolete and therefore unimportant"

It is not a modern efficient default, but it remains a clean reference for CNN depth and local-filter design.

## What To Check In Later Vision Papers

- Is the baseline VGG, VGG-BN, ResNet, or another backbone?
- Are parameter count and FLOPs controlled?
- Is the gain from depth, width, kernel design, normalization, or training recipe?
- Does the paper test transfer or only ImageNet classification?
- Does the architecture preserve enough spatial detail for dense prediction?
- Is the full classifier head used, or only convolutional features?
- Are pretrained weights used?

## Why It Still Matters

VGG gives a clean design lesson:

$$
\text{small kernels}
+
\text{more layers}
+
\text{simple repeated blocks}
\Rightarrow
\text{strong CNN backbone}.
$$

For an architecture wiki, it is the bridge from [[papers/architectures/alexnet|AlexNet]] to [[papers/architectures/deep-residual-learning|ResNet]]:

- AlexNet: deep CNNs can win on ImageNet.
- VGG: deeper, simpler CNNs with small filters are strong.
- ResNet: very deep CNNs need residual optimization paths.

## Limitations

- Parameter-heavy fully connected head.
- Inefficient compared with later CNN designs.
- Plain depth becomes hard to optimize without residual connections.
- Original design predates modern normalization and augmentation recipes.
- Classification-focused benchmark does not fully test dense prediction or visual reasoning.
- Local convolution bias may be limiting when global context is central.

## Connections

- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/convolution|Convolution]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/modalities/image|Image]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/inception|Inception]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/index|Architecture papers]]
