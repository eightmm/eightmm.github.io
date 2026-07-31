---
title: World Models
tags:
  - papers
  - architectures
  - generative-models
  - reinforcement-learning
  - agents
---

# World Models

> **One-line claim:** an agent can learn a compact visual latent state, model its dynamics with a recurrent mixture model, and optimize a controller inside the learned model rather than requiring every controller update to interact with the real environment.

## Citation

- Authors: David Ha and Jürgen Schmidhuber
- Year: 2018
- Paper: [World Models](https://arxiv.org/abs/1803.10122)
- Code and project material: linked from the arXiv record

## Why this paper belongs in Architecture Papers

World Models is a model-based agent architecture with three explicit interfaces:

$$
\text{observation}
\xrightarrow{\text{VAE}}
\text{latent state}
\xrightarrow{\text{MDN-RNN}}
\text{predicted dynamics}
\xrightarrow{\text{controller}}
\text{action}.
$$

The contribution is not simply “use a VAE in reinforcement learning.” It is the decomposition of an agent into a visual representation model, a probabilistic temporal model, and a controller that can be trained in the model's imagined environment. Each component has a distinct contract and can be replaced independently.

This makes the paper a useful companion to [[papers/architectures/human-level-control-through-deep-reinforcement-learning|DQN]]. DQN learns a value function from replayed real transitions. World Models attempts to learn a compact transition model and use imagined transitions for control.

## Problem setup

Let $x_t$ be an observation, $z_t$ a latent representation, $a_t$ an action, and $r_t$ a reward. A model-based agent wants to approximate the environment transition distribution:

$$
p(z_{t+1},r_t\mid z_t,a_t,h_t),
$$

where $h_t$ is a recurrent memory summarizing the history. The true environment remains the source of observations, but the learned model supplies additional hypothetical transitions during controller training.

The architectural question is therefore:

> What information must be preserved in the latent state for a controller to predict and influence future reward?

A latent representation that reconstructs images well is not automatically sufficient for control. It must preserve task-relevant information and support a useful predictive dynamics model.

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Visual encoder | observation $x_t$ | latent distribution $q_\phi(z_t\mid x_t)$ | compresses high-dimensional perception |
| Latent sampler | latent distribution | $z_t$ | provides a compact state for the dynamics model |
| MDN-RNN | $z_t$, $a_t$, recurrent state | distribution of $z_{t+1}$ and $r_t$ | predicts stochastic future outcomes |
| Controller | $z_t$, recurrent state | $a_t$ | maps the modeled state to an action |
| Environment | action | next observation and reward | supplies real transitions and evaluation |

The modules are coupled during data collection but can be trained on different objectives. The VAE learns visual compression, the MDN-RNN learns temporal prediction, and the controller optimizes reward under the chosen rollout procedure.

## The visual latent model

The visual model is a variational autoencoder. The encoder produces a distribution over latent variables:

$$
q_\phi(z_t\mid x_t)
=
\mathcal{N}\left(
\mu_\phi(x_t),
\operatorname{diag}(\sigma_\phi^2(x_t))
\right).
$$

The reparameterization trick makes sampling differentiable:

$$
z_t
=
\mu_\phi(x_t)
+
\sigma_\phi(x_t)\odot\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

A standard VAE objective can be written as:

$$
\mathcal{L}_{\mathrm{VAE}}
=
\mathbb{E}_{q_\phi(z\mid x)}[-\log p_\psi(x\mid z)]
+
\beta\,D_{\mathrm{KL}}
\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

The reconstruction term encourages the latent code to preserve visual information. The KL term regularizes the code toward a prior. The controller does not need every pixel detail, so the latent bottleneck is a deliberate architectural bias rather than only a compression trick.

The paper uses a compact visual state for control experiments. When implementing the idea, the exact latent dimensionality, decoder capacity, normalization, and frame preprocessing should be reported because they change the information available to the dynamics model.

## The recurrent dynamics model

The MDN-RNN receives the latent state and action and updates a recurrent hidden state:

$$
h_{t+1}
=
\operatorname{RNN}_\eta(h_t,z_t,a_t).
$$

Instead of predicting a single next latent vector, a mixture density network predicts a distribution. For a $K$-component Gaussian mixture:

$$
p(z_{t+1}\mid h_{t+1})
=
\sum_{k=1}^{K}
\pi_k(h_{t+1})
\mathcal{N}\left(
z_{t+1};
\mu_k(h_{t+1}),
\operatorname{diag}(\sigma_k^2(h_{t+1}))
\right),
$$

with

$$
\pi_k\ge 0,
\qquad
\sum_{k=1}^{K}\pi_k=1.
$$

The reward prediction can be modeled with a separate output head:

$$
\hat r_t=f_r(h_t),
$$

and a terminal or continuation signal can be predicted when the environment requires it. The mixture is important when several futures are plausible. A deterministic mean predictor can average incompatible outcomes and produce a latent state that never occurs in the real environment.

The training objective is the negative log-likelihood of observed latent transitions and rewards:

$$
\mathcal{L}_{\mathrm{RNN}}
=
-\sum_t
\log p_\eta(z_{t+1}\mid h_{t+1})
-
\lambda_r\sum_t
\log p_\eta(r_t\mid h_t).
$$

Teacher forcing uses the observed latent $z_t$ during training. During an imagined rollout, the model feeds sampled or predicted latents back into itself. This train/inference mismatch is one source of model error accumulation.

## The controller

The controller maps the latent state and recurrent memory to an action:

$$
a_t
=
\pi_\omega(z_t,h_t).
$$

For continuous actions, the controller may emit a bounded vector; for discrete actions, it may emit logits or a categorical choice. The original experiments use a compact controller so that optimization can focus on the quality of the learned world model.

The controller objective over a rollout of length $H$ is:

$$
J(\omega)
=
\mathbb{E}_{\hat{p}_\eta}
\left[
\sum_{t=0}^{H-1}\gamma^t\hat r_t
\right],
$$

where $\hat{p}_\eta$ denotes the learned transition distribution. In the real environment, performance is evaluated using the true transition process, not only the model's predicted reward.

## Dream rollouts

After learning the VAE and MDN-RNN from environment trajectories, the controller can be optimized through imagined trajectories:

$$
\hat z_{t+1}
\sim
p_\eta(\cdot\mid \hat z_t,a_t,\hat h_t),
\qquad
a_t=\pi_\omega(\hat z_t,\hat h_t).
$$

The word “dream” refers to this rollout inside the learned model. It does not mean the model has recovered a complete, faithful simulator. It means the controller receives a cheap source of hypothetical experience.

This changes the data-cost profile:

$$
\text{real interaction}
\rightarrow
\text{dataset}
\rightarrow
\text{world model}
\rightarrow
\text{many cheap imagined rollouts}.
$$

The benefit is largest when real interaction is expensive. The risk is largest when the controller exploits a prediction error that looks like high reward inside the model.

## What the experiments establish

The paper demonstrates the complete decomposition on visual control tasks, including VizDoom and CarRacing-style environments, and shows that a controller can be optimized using the learned model. The evidence supports the architectural feasibility of:

- compressing images to a low-dimensional latent state;
- predicting stochastic latent dynamics with an MDN-RNN;
- training a small controller with imagined rollouts;
- evaluating the resulting controller in the actual environment.

The paper does not establish that a learned model is always safer, more sample efficient, or more accurate than model-free reinforcement learning. The result depends on the quality of the latent representation, the prediction horizon, the controller optimizer, and how frequently the model is refreshed with real data.

## Ablation questions

- Does a deterministic RNN perform worse than a mixture density model when the future is multimodal?
- How much does visual reconstruction quality correlate with control quality?
- Does increasing latent dimension improve control, or only reconstruction?
- How quickly do errors compound when imagined rollouts become longer?
- Does controller optimization exploit model artifacts that disappear in the real environment?
- What happens when the VAE and dynamics model are updated online rather than in separate stages?
- How much real data is required before model-based controller training becomes useful?

These questions separate the three possible sources of improvement: representation, dynamics prediction, and controller optimization.

## Complexity and scaling

For a latent dimension $d_z$, hidden dimension $d_h$, action dimension $d_a$, and mixture count $K$, the dynamics output size grows approximately with the number of mixture parameters:

$$
O\left(K(2d_z+1)\right)
$$

for diagonal Gaussian means, scales, and mixture logits, before reward and terminal heads are added. A larger mixture count increases expressive capacity but also makes likelihood optimization and calibration more difficult.

Imagined rollout cost is approximately:

$$
O(H\cdot C_{\mathrm{RNN}})
$$

per rollout, where $H$ is the horizon. The model can be cheaper than rendering the real environment, but the number of sampled rollouts and controller evaluations can still dominate total compute.

## Relation to nearby architectures

| Paper or concept | Main difference |
| --- | --- |
| [DQN](/papers/architectures/human-level-control-through-deep-reinforcement-learning) | learns action values from real replayed transitions rather than a learned latent simulator |
| [VAE](/papers/architectures/auto-encoding-variational-bayes) | supplies the latent-variable representation component, but not the full agent architecture |
| [RNN encoder-decoder](/papers/architectures/rnn-encoder-decoder) | supplies temporal recurrence; World Models combines recurrence with probabilistic next-state prediction |
| [Agents](/agents) | World Models is an internal world-model/controller design, while agents also include planning, tools, memory, and external feedback loops |
| [Neural Turing Machines](/papers/architectures/neural-turing-machines) | provides differentiable external memory, not a learned environment transition model |
| [Generative models](/concepts/generative-models) | the MDN predicts future latent states, but the overall objective is control rather than unconstrained sample generation |

## Limits and failure modes

- Reconstruction loss can preserve visually salient information that is irrelevant to reward and discard small but control-critical signals.
- Probabilistic predictions can be well calibrated one step ahead but unusable over long horizons.
- A controller can exploit errors in the learned model, producing high imagined reward and poor real performance.
- Dataset coverage limits the model's ability to predict states outside the behavior distribution.
- The separation between VAE, dynamics model, and controller can make end-to-end credit assignment indirect.
- Evaluation on small environments does not prove robust world modeling for open-ended tasks.

## Implementation checklist

- [ ] Define the observation, latent, action, reward, and terminal contracts.
- [ ] Report the VAE latent dimension and whether sampling or the posterior mean is used.
- [ ] Evaluate latent reconstruction and downstream control separately.
- [ ] Measure one-step and multi-step dynamics prediction error.
- [ ] Compare deterministic and probabilistic dynamics models.
- [ ] Track real versus imagined return during controller optimization.
- [ ] Evaluate controllers in the real environment without selecting only favorable seeds.
- [ ] Bound imagined rollout horizon and refresh the model with new real data when needed.

## Takeaway

World Models is a canonical example of a **model-based agent architecture**: perception creates a compact state, a recurrent probabilistic model predicts what actions may do, and a controller acts inside that learned state space. Its durable lesson is the interface between representation, dynamics, and control, together with the need to measure model error separately from policy quality.

## Related notes

- [[ai/architectures|Architectures]]
- [[ai/generative-models|Generative models]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[agents/index|Agents]]
- [[papers/architectures/human-level-control-through-deep-reinforcement-learning|Human-level control through deep reinforcement learning]]
