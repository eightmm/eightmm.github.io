---
title: Dream to Control Learning Behaviors by Latent Imagination
tags:
  - papers
  - architectures
  - agents
  - reinforcement-learning
  - world-models
---

# Dream to Control: Learning Behaviors by Latent Imagination

> **One-line claim:** Dreamer learns a compact latent world model from images and improves a policy by backpropagating value gradients through imagined latent trajectories instead of learning only from direct environment interaction.

## Citation

- Authors: Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi
- Year: 2019 (revised 2020)
- Paper: [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
- arXiv: [1912.01603](https://arxiv.org/abs/1912.01603)

## Why this paper belongs in Architecture Papers

Dreamer is an agent architecture whose main contribution is the boundary between a learned world model and a controller. Its policy is not trained by repeatedly rendering imagined pixels or by treating the model as a black-box simulator. Instead, the agent rolls forward in a compact latent state and differentiates through the learned dynamics.

The high-level dataflow is:

$$
o_t
\xrightarrow{\text{encoder}}
(h_t,z_t)
\xrightarrow{\text{RSSM}}
(h_{t+1},z_{t+1})
\xrightarrow{\text{actor/value}}
(a_t,v_t).
$$

There are two distinct loops:

$$
\text{real observations}
\rightarrow
\text{world-model learning}
$$

and

$$
\text{latent start states}
\rightarrow
\text{imagined rollout}
\rightarrow
\text{actor/value learning}.
$$

This separation makes Dreamer a useful architectural bridge between [[papers/architectures/world-models|World Models]], which uses a latent model and a separate controller, and [[papers/architectures/mastering-chess-and-shogi-by-self-play|AlphaZero]], which uses explicit tree search rather than differentiable latent rollouts.

## Problem setup

At time $t$, the environment emits an observation $o_t$, the agent chooses an action $a_t$, and the environment returns a reward $r_t$ and the next observation. The observation may be an image, so directly predicting pixels over a long horizon is expensive and often unnecessary for control.

The agent instead learns a latent state $s_t$ that should satisfy two requirements:

1. it must retain information useful for reconstructing or predicting observations;
2. it must support prediction of rewards and future states under actions.

The control objective is the discounted return:

$$
J(\pi)
=
\mathbb{E}_{\pi,p}
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right],
$$

where $\pi$ is the policy, $p$ is the environment transition process, and $\gamma$ is the discount factor. Dreamer estimates this objective using the learned latent transition model rather than requiring all policy updates to use $p$ directly.

## Architecture contract

| Component | Input | Output | Architectural role |
| --- | --- | --- | --- |
| Observation encoder | $o_t$ | posterior parameters for $z_t$ | compresses high-dimensional observations |
| Deterministic recurrent state | previous state and action | $h_t$ | stores action-conditioned history |
| Stochastic latent state | $h_t$ and observation evidence | $z_t$ | represents uncertainty and local observation information |
| RSSM prior | $h_t$ and previous latent/action | prior over $z_t$ | predicts latent futures without seeing the next observation |
| Observation decoder | $h_t,z_t$ | $\hat o_t$ | trains the representation model |
| Reward decoder | $h_t,z_t$ | $\hat r_t$ | makes the latent state useful for control |
| Continuation decoder | $h_t,z_t$ | $\hat c_t$ | estimates whether an episode continues |
| Actor | latent state | action distribution | chooses actions in imagined rollouts |
| Critic | latent state | value estimate | supplies a differentiable return target |

The central design is the Recurrent State-Space Model (RSSM). It combines a deterministic recurrent state with a stochastic latent variable rather than asking either a plain RNN or an independent latent code to carry the whole world state.

## RSSM state decomposition

Let $h_t$ denote deterministic memory and $z_t$ denote the stochastic part of the latent state. The recurrent transition is:

$$
h_t
=
f_\theta(h_{t-1},z_{t-1},a_{t-1}).
$$

After computing $h_t$, the model has two distributions over $z_t$:

$$
p_\theta(z_t\mid h_t)
\qquad\text{and}\qquad
q_\phi(z_t\mid h_t,o_t).
$$

The first is the prior used for imagination. It predicts what the next latent state could be from history and action alone. The second is the posterior or representation model. It additionally observes $o_t$ and is used while learning from real trajectories.

The complete latent state is often written as:

$$
s_t=(h_t,z_t).
$$

This factorization is important because imagined rollouts can sample from $p_\theta(z_t\mid h_t)$ without access to future observations, while training can use $q_\phi(z_t\mid h_t,o_t)$ to infer a state from data.

## World-model objective

For a real sequence of observations, actions, and rewards, the world model is trained to reconstruct observations, predict rewards and continuation, and make the prior agree with the observation-conditioned posterior.

A compact form of the sequence objective is:

$$
\mathcal{L}_{\mathrm{model}}
=
\sum_t
\Big[
-\log p_\theta(o_t\mid h_t,z_t)
-\log p_\theta(r_t\mid h_t,z_t)
-\log p_\theta(c_t\mid h_t,z_t)
\Big]
+
\beta\,D_{\mathrm{KL}}
\left(
q_\phi(z_t\mid h_t,o_t)
\,\middle\|\,
p_\theta(z_t\mid h_t)
\right).
$$

Here $c_t$ is a continuation or non-terminal target. The first three terms define what the latent state must predict. The KL term makes the prior usable during imagination.

The KL term is not merely a generic VAE regularizer. Its architectural role is to connect two interfaces:

$$
\text{posterior with real observation}
\longleftrightarrow
\text{prior without future observation}.
$$

If the posterior learns information that the prior cannot predict, the representation may reconstruct well while imagined control collapses. In implementation, KL balancing and stop-gradient choices matter because they determine whether the prior or posterior dominates this interface.

## Observation representation

For an image observation, the encoder produces parameters of the stochastic posterior. A categorical latent can be represented as a collection of logits:

$$
q_\phi(z_t\mid h_t,o_t)
=
\prod_{i=1}^{N}
\operatorname{Cat}(z_{t,i};
\operatorname{softmax}(\ell_{t,i})).
$$

The precise latent parameterization is less important than the contract: the encoder must produce a state that can be decoded and predicted, while the prior must be able to generate comparable states during imagination.

The observation decoder is trained with a likelihood appropriate to the observation type. For an image, a simple reconstruction term may be expressed as:

$$
\mathcal{L}_{\mathrm{obs}}
=
-\log p_\theta(o_t\mid h_t,z_t).
$$

Reconstruction is a training signal, not the final control metric. A visually faithful latent can still discard reward-relevant information, and a latent that reconstructs imperfectly can be sufficient for control.

## Latent imagination

After an observed sequence supplies a starting latent state $s_\tau=(h_\tau,z_\tau)$, Dreamer predicts future states without receiving future observations:

$$
h_{t+1}
=
f_\theta(h_t,z_t,a_t),
\qquad
z_{t+1}
\sim
p_\theta(z_{t+1}\mid h_{t+1}).
$$

The actor supplies the action:

$$
a_t\sim\pi_\psi(a_t\mid h_t,z_t).
$$

The reward and continuation models then provide predicted consequences:

$$
\hat r_t
\sim
p_\theta(r_t\mid h_t,z_t),
\qquad
\hat c_t
\sim
p_\theta(c_t\mid h_t,z_t).
$$

The resulting rollout is entirely in latent space:

$$
(h_\tau,z_\tau)
\rightarrow
(h_{\tau+1},z_{\tau+1})
\rightarrow
\cdots
\rightarrow
(h_{\tau+H},z_{\tau+H}).
$$

No predicted image needs to be rendered at each step. This is the main computational and optimization distinction from a pixel-space simulator.

## Actor and critic

The critic approximates the value of a latent state:

$$
v_\xi(s_t)
\approx
\mathbb{E}\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid s_t
\right].
$$

For an imagined trajectory, a bootstrapped value target can be written using a return estimate $V_t^{\lambda}$:

$$
V_t^{\lambda}
=
\hat r_t
+
\gamma\hat c_t
\left[
(1-\lambda)v_\xi(s_{t+1})
+
\lambda V_{t+1}^{\lambda}
\right].
$$

The continuation prediction gates future value. If the model predicts that an episode terminates, later rewards should not be propagated through that branch.

The actor is trained to increase imagined return:

$$
\mathcal{L}_{\mathrm{actor}}
=
-\mathbb{E}_{\pi_\psi,p_\theta}
\left[
\sum_{t=\tau}^{\tau+H}
\gamma^{t-\tau}
V_t^{\lambda}
\right].
$$

Because the transition is differentiable with respect to its inputs and parameters, gradients can flow through the imagined latent states:

$$
\frac{\partial V_t^{\lambda}}{\partial \psi}
=
\sum_{k\ge t}
\frac{\partial V_t^{\lambda}}{\partial s_k}
\frac{\partial s_k}{\partial a_{k-1}}
\frac{\partial a_{k-1}}{\partial\psi}.
$$

This is the architectural reason Dreamer differs from a policy that merely samples imagined episodes and applies a black-box policy-gradient estimator.

## Training loop

The complete loop alternates between real interaction and latent learning:

1. collect episodes with the current actor in the real environment;
2. add observations, actions, rewards, and continuation labels to replay;
3. infer posterior latent states from replay sequences;
4. update the RSSM, decoders, and representation using the world-model objective;
5. start imagined rollouts from replay states;
6. update actor and critic on those rollouts;
7. periodically evaluate the actor in the real environment.

The distinction between data sources is essential:

| Update | Data source | What it teaches |
| --- | --- | --- |
| World model | real replay sequences | what observations and actions imply |
| Critic | imagined latent trajectories | value of model-predicted futures |
| Actor | imagined latent trajectories | actions that improve predicted return |
| Evaluation | real environment | whether the learned policy actually works |

Imagined data can increase the number of policy-training transitions, but it cannot replace coverage from real data. The world model's support is bounded by the states represented in replay and by the accuracy of its learned transition distribution.

## What the experiments establish

The paper evaluates Dreamer on 20 visual control tasks and reports improvements in data efficiency, computation time, and final performance over the compared approaches. The paper's architectural evidence supports the following claims:

- a compact latent state can be sufficient for long-horizon visual control;
- a recurrent stochastic model can provide useful imagined trajectories;
- analytic value gradients through latent trajectories can train a policy;
- one world-model/controller design can operate across multiple visual control tasks.

These results do not prove that latent imagination is always superior to model-free reinforcement learning. They also do not imply that low reconstruction error guarantees good control. The strongest conclusion is narrower: a learned latent transition model can be used as a differentiable training environment when its representation and predictive state are adequate.

## Ablation questions

- Does the stochastic latent variable improve control when observations are partially observable or futures are multimodal?
- How much performance comes from the deterministic recurrent state versus the stochastic state?
- Does training the actor on shorter imagined horizons avoid model exploitation at the cost of myopic behavior?
- How sensitive are results to KL balancing, latent capacity, and the posterior-prior parameterization?
- Does reward prediction matter more than observation reconstruction for downstream control?
- How does imagined-data quality change as the replay distribution shifts under the improving policy?
- Does a critic trained only in imagination become miscalibrated on real states?
- What is the fair compute comparison when imagined rollout count, model updates, and real environment steps are all reported?

These questions separate representation quality, transition quality, optimization method, and data efficiency instead of attributing every gain to “world models.”

## Failure modes

### Model exploitation

The actor may find actions that produce high predicted reward because of a model error. The predicted return can then increase while real-environment return decreases.

$$
\hat J(\pi)\gg J_{\mathrm{real}}(\pi)
$$

The gap should be monitored rather than hidden inside an aggregate training curve.

### Long-horizon error accumulation

One-step prediction quality does not imply accurate multi-step rollouts:

$$
p_\theta(s_{t+H}\mid s_t,a_{t:t+H-1})
\ne p(s_{t+H}\mid s_t,a_{t:t+H-1})
$$

for sufficiently large $H$. Short imagined horizons can reduce compounding error, while long horizons provide more planning signal. The choice is an architecture-training trade-off.

### Representation mismatch

The posterior may encode details visible in $o_t$ that the prior cannot predict, or the decoder may reward pixel fidelity that is irrelevant to the task. The relevant question is not whether the latent reconstructs every detail, but whether it is predictive and controllable.

### Dataset and coverage limits

Replay contains only states reached by the current data-collection policy. A model trained on narrow behavior may be confidently wrong outside that support. Exploration, replay composition, and uncertainty estimation are therefore part of the practical system contract.

## Comparison with nearby architectures

| Paper or concept | Architectural difference |
| --- | --- |
| [World Models](/papers/architectures/world-models) | uses a VAE, MDN-RNN, and controller; Dreamer makes latent actor/value learning and backpropagation through imagined dynamics the central training mechanism |
| [DQN](/papers/architectures/human-level-control-through-deep-reinforcement-learning) | learns an action-value function from real replay and does not require a learned transition model |
| [AlphaZero](/papers/architectures/mastering-chess-and-shogi-by-self-play-with-a-general-reinforcement-learning-algorithm) | improves a policy with explicit tree search; Dreamer uses differentiable latent rollouts rather than per-decision MCTS |
| [Auto-Encoding Variational Bayes](/papers/architectures/auto-encoding-variational-bayes) | provides latent-variable inference machinery, but not recurrent action-conditioned dynamics or control learning |
| [Reinforcement learning](/concepts/learning/reinforcement-learning) | supplies the objective family; Dreamer is a particular world-model, actor, critic, and imagination architecture |
| [Generative models](/ai/generative-models) | the RSSM models future latent states conditionally; the complete system is optimized for control, not unconditional sample quality |
| [Agents](/agents) | Dreamer describes environment-facing control architecture, while tool-using agents add memory, tools, planning, and verification interfaces |

## Reproduction checklist

- [ ] Record observation preprocessing, action space, reward scaling, and continuation semantics.
- [ ] Specify deterministic and stochastic latent dimensions separately.
- [ ] Report the RSSM prior and posterior parameterization and KL balancing rule.
- [ ] Measure reconstruction, reward prediction, continuation prediction, and multi-step latent prediction separately.
- [ ] State the imagined horizon, discount, lambda-return setting, and actor/value update ratio.
- [ ] Compare imagined return with real-environment return throughout training.
- [ ] Keep real environment steps and model-training compute as separate budgets.
- [ ] Evaluate multiple seeds and report failures caused by model exploitation or unstable rollout prediction.
- [ ] Compare against a model-free baseline under the same real-interaction budget.

## Takeaway

Dreamer makes a precise architectural proposal: learn a latent state that is both inferable from observations and predictable under actions, then use that state as a differentiable training space for actor and value learning. The durable lesson is the interface between **representation**, **dynamics**, and **control**. Latent imagination is useful only when those interfaces remain aligned with real-environment outcomes.

## Related notes

- [[ai/architectures|Architectures]]
- [[papers/architectures/world-models|World Models]]
- [[papers/architectures/mastering-chess-and-shogi-by-self-play|AlphaZero]]
- [[papers/architectures/human-level-control-through-deep-reinforcement-learning|DQN]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[ai/generative-models|Generative models]]
- [[agents/index|Agents]]
