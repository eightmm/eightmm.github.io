---
title: Neural Architecture Search with Reinforcement Learning
tags:
  - papers
  - architectures
  - neural-architecture-search
  - reinforcement-learning
---

# Neural Architecture Search with Reinforcement Learning

> **One-line claim:** a recurrent controller can generate neural-network descriptions and use validation accuracy as a reward to search over architectures instead of relying entirely on manual design.

## Citation

- Authors: Barret Zoph and Quoc V. Le
- Year: 2016 preprint; 2017 revision
- Paper: [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)

## Why this paper belongs in Architecture Papers

Most architecture papers define one model. NAS papers define a **procedure that generates and evaluates models**. The object being studied is therefore a two-level system:

$$
\text{controller}
\xrightarrow{\text{architecture tokens}}
\text{child network}
\xrightarrow{\text{validation training}}
\text{reward}.
$$

The paper belongs on the architecture shelf because the search space, controller output, child-network interface, and evaluation budget determine which architectures can be discovered. It should not be read as evidence that reinforcement learning is automatically the best way to design any model.

## Problem setup

Let $m$ denote a model description sampled from a search space $\mathcal{M}$. The model is trained on a training split and evaluated on a validation split. The controller wants to maximize expected validation reward:

$$
J(\theta)
=
\mathbb{E}_{m\sim\pi_\theta(m)}
\left[R(m)\right],
$$

where $\pi_\theta$ is the controller distribution and $R(m)$ is typically validation accuracy or a related score.

The search process has two distinct learning problems:

| Level | Learner | Optimized signal |
| --- | --- | --- |
| Inner loop | child-network weights $w$ | training loss on data |
| Outer loop | controller parameters $\theta$ | validation reward of sampled architecture |

The child model is not differentiated through the full training process in the original policy-gradient formulation. The controller receives a scalar reward after a child model has been trained.

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Search space | constraints and primitive operations | valid architecture descriptions | defines what can be discovered |
| Controller RNN | previous architecture tokens | distribution over next tokens | samples model descriptions |
| Child network builder | architecture description | executable network | materializes one candidate |
| Inner training loop | child network and training data | trained child weights | estimates candidate quality |
| Validation evaluator | trained child and validation data | scalar reward | supplies outer-loop signal |
| Policy update | sampled descriptions and rewards | new controller parameters | increases probability of high-reward descriptions |

The separation between search space and controller is essential. If the search space excludes a useful operation, no amount of controller training can discover it. If the evaluator is noisy or overfit, the controller will optimize that artifact.

## Controller as an architecture generator

The controller is an RNN that emits a sequence of discrete decisions:

$$
m=(a_1,a_2,\ldots,a_T).
$$

Its probability factorizes autoregressively:

$$
\pi_\theta(m)
=
\prod_{t=1}^{T}
\pi_\theta(a_t\mid a_{<t}).
$$

Each token may specify an operation, filter size, number of filters, activation, connectivity choice, or recurrent-cell component. The precise token vocabulary is part of the paper's search space rather than a universal NAS language.

The controller's hidden state summarizes earlier decisions:

$$
h_t
=
\operatorname{RNN}_\theta(h_{t-1},a_{t-1}),
$$

and the next decision distribution is:

$$
\pi_\theta(a_t\mid a_{<t})
=
\operatorname{softmax}(W_oh_t+b_o).
$$

The controller is not the final predictor. It is a policy over programs or graph descriptions that instantiate child predictors.

## Child architecture

For a sampled description $m$, a builder creates a child network:

$$
f_{m,w}:x\mapsto\hat y.
$$

The child weights are trained by minimizing a task loss:

$$
w_m^*
=
\arg\min_w
\mathcal{L}_{\mathrm{train}}(f_{m,w}).
$$

The outer reward is then computed:

$$
R(m)
=
\operatorname{Score}
\left(f_{m,w_m^*};D_{\mathrm{valid}}\right).
$$

The notation hides a major practical approximation: the child is rarely trained to full convergence for every candidate. Early stopping, reduced data, weight sharing, proxy tasks, and low-fidelity training all change the reward estimator.

## Policy-gradient update

Because the reward arrives after discrete architecture decisions and child training, the controller can use the score-function estimator:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{m\sim\pi_\theta}
\left[
R(m)\nabla_\theta\log\pi_\theta(m)
\right].
$$

With sampled architectures $m_k$ and a baseline $b$:

$$
\widehat{\nabla_\theta J}
=
\frac{1}{K}
\sum_{k=1}^{K}
\left(R(m_k)-b\right)
\nabla_\theta\log\pi_\theta(m_k).
$$

The baseline reduces variance without changing the expected gradient when it does not depend on the sampled action in the relevant way. The controller increases probability for decisions that produced above-baseline validation reward.

The update is noisy because the same architecture can receive different rewards from random initialization, data order, augmentation, and partial training. Reward normalization and repeated evaluation are therefore part of a defensible NAS experiment.

## Search spaces

A search space can be represented as a graph of decisions:

$$
\mathcal{M}
=
\mathcal{O}_1
\times
\mathcal{O}_2
\times\cdots\times
\mathcal{O}_T,
$$

where each $\mathcal{O}_t$ is a set of allowed operations or hyperparameters. In practice, validity constraints couple decisions, so the Cartesian product is only an approximation.

Typical choices include:

| Search dimension | Examples | Consequence |
| --- | --- | --- |
| operation | convolution, pooling, identity, recurrent transform | changes inductive bias |
| kernel or receptive field | $1\times1$, $3\times3$, dilated | changes locality and cost |
| width | channel count, hidden size | changes capacity and memory |
| depth | number of blocks or cells | changes optimization and latency |
| connectivity | skip, branch, merge, cell edge | changes information flow |
| activation or normalization | ReLU, gated unit, normalization choice | changes stability and expressivity |
| recurrent cell choice | gate and update primitives | changes state dynamics |

The search space is often more important than the controller. A simple controller over a well-designed space can outperform a sophisticated controller over a badly constrained space.

## Cell-based search

To reduce the number of decisions, the paper searches for a reusable cell and stacks copies of it in a larger network. A cell can be viewed as a directed acyclic computation graph:

$$
h_j
=
\sum_{i<j}
o_{i,j}(h_i),
$$

where $o_{i,j}$ is a selected operation on an edge. The macro-network then repeats the discovered cell under a manually specified stem, reduction schedule, or number of cells.

This creates a useful but important boundary:

- **cell search** optimizes a local computation motif;
- **macro architecture** determines how cells are arranged at scale.

A cell that is strong in one depth, resolution, or data regime may not remain strong after stacking. The transfer step must be evaluated rather than assumed.

## Reward design

The simplest reward is validation accuracy:

$$
R(m)=\operatorname{Acc}_{\mathrm{valid}}(m).
$$

For deployment, a multi-objective reward may be more appropriate:

$$
R(m)
=
\operatorname{Acc}(m)
-
\lambda_t\log\operatorname{Latency}(m)
-
\lambda_p\log\operatorname{Params}(m)
-
\lambda_e\operatorname{Energy}(m).
$$

The terms must be measured under the intended hardware and batch regime. Parameter count is not a substitute for latency, and FLOPs are not a substitute for memory traffic or kernel availability.

If the reward is noisy:

$$
R(m)=\mu(m)+\varepsilon,
\qquad
\mathbb{E}[\varepsilon]=0,
$$

then the controller may still over-select candidates with lucky high observations. Repeated seeds, confidence intervals, and final retraining are needed before calling an architecture superior.

## What the experiments establish

The paper reports a controller that generates CNN architectures for CIFAR-10 and recurrent cells for Penn Treebank. The arXiv abstract reports a CIFAR-10 test error of 3.65 and a Penn Treebank perplexity of 62.4 for the reported configurations.

The narrower architectural contribution is the demonstration that an outer-loop controller can generate structured model descriptions and optimize them with validation reward. The results do not establish that RL-based NAS is cheaper than manual design, random search, evolutionary methods, or differentiable NAS in general.

## Ablation questions

- How much does the controller improve over random sampling from the same search space?
- What is the quality distribution of sampled architectures, not only the best one?
- How does the validation reward change when child models receive more or less training?
- Does the discovered cell transfer across depth, width, resolution, and dataset?
- Are the reported gains due to the search method or to a favorable search space?
- How many total child-model updates and accelerator hours were used?
- Does the final architecture remain strong after independent retraining with fresh seeds?
- How sensitive is the policy to reward normalization, baseline, and controller entropy?

The correct comparison is a **search-budget comparison**. Comparing only the final model's accuracy hides the cost of discovering it.

## Complexity and scaling

If one candidate requires $C_{\mathrm{child}}$ training cost and the controller evaluates $N$ candidates, the naive search cost is approximately:

$$
C_{\mathrm{NAS}}
\approx
N\cdot C_{\mathrm{child}}
+
C_{\mathrm{controller}}.
$$

This is why later NAS systems introduce weight sharing, low-fidelity proxies, early stopping, surrogate models, or differentiable relaxation. Each reduces cost by introducing a bias into the reward estimate.

The architecture itself may be cheap at inference while the search is extremely expensive. A paper should report both:

$$
\text{deployment cost}
\neq
\text{search cost}.
$$

## Relation to nearby architectures and methods

| Paper or concept | Main difference |
| --- | --- |
| [EfficientNet](/papers/architectures/efficientnet) | uses NAS-derived EfficientNet-B0, then proposes compound scaling as the main architecture rule |
| [Mixture of Experts](/papers/architectures/sparsely-gated-moe) | routes inputs among experts at inference; NAS routes a controller through architecture decisions before training the final model |
| [Reinforcement learning](/concepts/learning/reinforcement-learning) | supplies the outer-loop optimization framework, not the search space or child-network contract |
| [Universal Transformers](/papers/architectures/universal-transformers) | a fixed model architecture with shared depth recurrence, not a procedure for generating architectures |
| [Architecture selection](/concepts/architectures/architecture-selection) | reusable decision framework for choosing a model; NAS automates part of that search under a defined objective |
| [Computational complexity](/concepts/architectures/computational-complexity) | provides the cost language needed to define a realistic NAS reward |

## Limits and failure modes

- The controller cannot discover architectures excluded by the search space.
- Validation reward can be overfit through repeated adaptive search.
- Child-model stochasticity makes the outer reward noisy and expensive.
- Partial-training proxies can rank architectures differently from full training.
- Search cost may exceed the value of the resulting architecture.
- Cell transfer can fail when the macro-network changes scale or data regime.
- Accuracy-only reward can select models that are unusable under memory, latency, or energy constraints.
- Reproducibility requires the search space, controller seed, budget, hardware, and final retraining protocol.

## Implementation checklist

- [ ] Write the search space and validity constraints before training the controller.
- [ ] Define the reward, validation split, fidelity, and compute budget explicitly.
- [ ] Compare against random search in the same space and budget.
- [ ] Log every sampled architecture and its training/evaluation conditions.
- [ ] Measure latency and memory on the target hardware, not only parameter count.
- [ ] Retrain selected architectures independently with multiple seeds.
- [ ] Separate architecture-search cost from final deployment cost.
- [ ] Report whether cell-level findings transfer to the final macro-network.

## Takeaway

Neural Architecture Search with Reinforcement Learning turns architecture design into a nested optimization problem: a controller generates a candidate, a child network is trained, and validation evidence updates the controller. Its durable lesson is the explicit contract between search space, candidate model, reward, and budget. The controller is only as meaningful as those four choices.

## Related notes

- [[ai/architectures|Architectures]]
- [[concepts/architectures/architecture-search|Architecture search]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[papers/architectures/efficientnet|EfficientNet]]
- [[papers/architectures/sparsely-gated-moe|Sparsely-Gated Mixture-of-Experts]]
