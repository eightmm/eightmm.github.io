---
title: Highway Networks
aliases:
  - papers/highway-networks
  - papers/highway-network
tags:
  - papers
  - architectures
  - residual
  - gating
---

# Highway Networks

> The paper introduced gated skip paths for very deep feed-forward networks, making depth easier to optimize before residual networks became the dominant simplified form.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Highway Networks |
| Authors | Rupesh Kumar Srivastava, Klaus Greff, Jurgen Schmidhuber |
| Year | 2015 |
| Venue | arXiv extended abstract; related NeurIPS 2015 paper as Training Very Deep Networks |
| arXiv | [1505.00387](https://arxiv.org/abs/1505.00387) |
| Related paper | [Training Very Deep Networks, arXiv 1507.06228](https://arxiv.org/abs/1507.06228) |
| Status | full note started |

## Question

Before [[papers/architectures/deep-residual-learning|ResNet]], very deep feed-forward networks were hard to train. Adding more layers often increased optimization difficulty rather than improving representation quality.

The paper asks:

$$
\text{Can a feed-forward network learn when to transform a representation and when to pass it through?}
$$

The idea is borrowed from gated recurrence: use gates to regulate information flow, but apply them across depth rather than time.

## Main Claim

Highway networks add learned transform and carry gates to each layer:

$$
y
=
H(x, W_H)\odot T(x, W_T)
+
x\odot C(x, W_C).
$$

The common coupled-gate form sets:

$$
C(x,W_C)=1-T(x,W_T),
$$

so the layer becomes:

$$
y
=
H(x)\odot T(x)
+
x\odot (1-T(x)).
$$

where:

| Symbol | Meaning |
| --- | --- |
| $x$ | input activation to the layer |
| $H(x)$ | nonlinear transform branch |
| $T(x)$ | transform gate |
| $C(x)$ | carry gate |
| $\odot$ | elementwise product |

The durable architecture claim is:

$$
\text{deep feed-forward computation}
+
\text{learned carry paths}
\Rightarrow
\text{easier optimization at large depth}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | activation vector or feature map $x$ |
| Output | activation of the same shape as $x$ |
| Main branch | nonlinear transformation $H(x)$ |
| Skip branch | carried input $x$ |
| Gate | input-dependent transform/carry gate |
| Depth role | allow information and gradients to cross many layers |
| Shape constraint | $H(x)$, $T(x)$, and $x$ must be broadcast-compatible |
| Later simplified form | ungated additive residual block |

A highway layer is not only:

$$
y = H(x).
$$

It is a mixture of transforming and carrying:

$$
y_i
=
T_i(x)H_i(x)
+
(1-T_i(x))x_i.
$$

Each dimension can choose a different update strength.

## Gate Parameterization

The transform gate is usually sigmoid:

$$
T(x)=\sigma(W_Tx+b_T).
$$

Then:

$$
0 < T_i(x) < 1.
$$

If $T_i(x)\approx 1$:

$$
y_i \approx H_i(x),
$$

so the layer transforms that feature.

If $T_i(x)\approx 0$:

$$
y_i \approx x_i,
$$

so the layer carries that feature through.

This makes the layer an input-dependent interpolation:

$$
y
=
x
+
T(x)\odot (H(x)-x).
$$

That form is useful because it makes the relationship to residual updates explicit.

## Relation To Residual Networks

The ResNet block is:

$$
y=x+F(x).
$$

A highway layer in update form is:

$$
y=x+T(x)\odot(H(x)-x).
$$

So the highway layer can be read as a gated residual-like update:

$$
F_{\text{highway}}(x)=T(x)\odot(H(x)-x).
$$

The important difference:

| Aspect | Highway Network | ResNet |
| --- | --- | --- |
| Skip path | gated carry path | mostly identity addition |
| Update strength | learned input-dependent gate | implicit through residual branch |
| Extra parameters | transform/carry gate projections | no gate in the basic block |
| Optimization bias | can learn to carry or transform | identity path always open |
| Later adoption | less common as a default block | became the dominant deep backbone pattern |

ResNet can be viewed as the simpler, more widely adopted route: keep the identity path open and learn an additive residual correction.

## Relation To LSTM

The paper explicitly follows the same broad idea as [[papers/architectures/long-short-term-memory|LSTM]]: use gates to protect information flow.

LSTM across time:

$$
c_t
=
f_t\odot c_{t-1}
+
i_t\odot \tilde{c}_t.
$$

Highway network across depth:

$$
x_{\ell+1}
=
C_\ell(x_\ell)\odot x_\ell
+
T_\ell(x_\ell)\odot H_\ell(x_\ell).
$$

The analogy is:

| LSTM | Highway Network |
| --- | --- |
| time step $t$ | layer depth $\ell$ |
| cell state $c_t$ | carried activation $x_\ell$ |
| input/forget gates | transform/carry gates |
| recurrent memory flow | feed-forward information flow |

The core pattern is the same:

$$
\text{new state}
=
\text{keep gate}\cdot \text{old state}
+
\text{write gate}\cdot \text{new content}.
$$

## Why It Helps Optimization

For a plain deep network:

$$
x_{\ell+1}=H_\ell(x_\ell).
$$

Backpropagation passes through repeated Jacobians:

$$
\frac{\partial \mathcal{L}}{\partial x_\ell}
=
\frac{\partial \mathcal{L}}{\partial x_L}
\prod_{k=\ell}^{L-1}
\frac{\partial x_{k+1}}{\partial x_k}.
$$

If those Jacobians repeatedly shrink or amplify signals, optimization becomes unstable.

With a highway layer:

$$
x_{\ell+1}
=
x_\ell
+
T_\ell(x_\ell)\odot(H_\ell(x_\ell)-x_\ell).
$$

When the gate leans toward carry behavior, the derivative includes an identity-like route. This creates shorter effective paths through depth:

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
\approx
I
\text{controlled correction}.
$$

The mechanism does not guarantee better generalization. It makes very deep networks easier to optimize.

## Bias Initialization

The gate bias matters. If transform gates start too open, a deep highway network behaves closer to a plain deep network:

$$
T(x)\approx 1
\Rightarrow
y\approx H(x).
$$

If transform gates start more closed, the network initially carries information:

$$
T(x)\approx 0
\Rightarrow
y\approx x.
$$

This resembles forget-gate bias choices in recurrent networks. The model starts with an easier information highway and learns where transformation is useful.

## Evidence To Read Carefully

The paper reports that very deep highway networks can be trained directly with stochastic gradient descent, while comparable plain networks become difficult as depth increases.

For architecture reading, split the evidence:

| Claim | Evidence Type | Caution |
| --- | --- | --- |
| Gates ease optimization | trainability of much deeper networks | optimization gain is not the same as universal accuracy gain |
| Carry paths help gradient flow | deep models do not degrade like plain counterparts | initialization and gate bias matter |
| Depth becomes experimentally accessible | hundreds of layers can be explored | later ResNet made a simpler variant dominant |

The paper should be read as a depth-optimization architecture paper, not as a final backbone recipe for modern vision or language models.

## Failure Modes

| Failure Mode | Mechanism | Practical Check |
| --- | --- | --- |
| Gate saturation | sigmoid gates become near 0 or 1 too early | inspect transform-gate statistics |
| Over-carrying | many layers pass input through with little transformation | check residual/update branch norms |
| Extra parameter cost | each layer needs gate projections | compare to ungated residual block |
| Shape mismatch | carry path requires compatible dimensions | use projection only when dimensions change |
| Over-crediting depth | deeper model may win due to parameter count or training time | compare matched compute and parameter budgets |

## Where It Fits

| Axis | Placement |
| --- | --- |
| Architecture family | gated feed-forward deep network |
| Core operation | transform/carry interpolation |
| Main concept | [residual connection](/concepts/architectures/residual-connection) |
| Predecessor idea | [LSTM](/papers/architectures/long-short-term-memory) gated information flow |
| Later simplified form | [ResNet](/papers/architectures/deep-residual-learning) |
| Modern echo | gated residual branches, GLU/SwiGLU blocks, residual stream scaling |

## Practical Checks

When reading later deep architecture papers, ask:

| Question | Why It Matters |
| --- | --- |
| Is the skip path identity, projected, gated, or scaled? | determines how easily information crosses depth |
| Is the gate input-dependent or a learned scalar? | changes whether routing is per-example or static |
| Are residual branches initialized small? | affects early optimization stability |
| Does the paper separate trainability from generalization? | deep trainability alone is not the final metric |
| Does the block preserve shape? | skip addition/carrying requires compatible tensors |

## Related

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/lstm|LSTM]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
