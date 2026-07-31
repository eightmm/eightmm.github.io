---
title: Mastering Atari Go Chess and Shogi by Planning with a Learned Model
tags:
  - papers
  - architectures
  - agents
  - reinforcement-learning
  - world-models
  - planning
---

# Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

> **One-line claim:** MuZero combines tree search with a learned latent model that predicts only the reward, policy, and value information needed for planning, without being given the environment's underlying dynamics.

## Citation

- Authors: Julian Schrittwieser et al.
- Year: 2019 (revised 2020)
- Paper: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
- Related journal publication: [Nature, 2020](https://doi.org/10.1038/s41586-020-03051-4)

## Why this paper belongs in Architecture Papers

MuZero is a canonical architecture for combining a learned model with explicit planning. It extends the policy/value network and Monte Carlo Tree Search pattern from [[papers/architectures/mastering-chess-and-shogi-by-self-play|AlphaZero]], but removes the requirement that the agent be given a perfect simulator or the rules of the environment.

The key change is not “learn a complete simulator.” MuZero learns a latent model whose outputs are sufficient for search:

$$
\text{observation history}
\xrightarrow{h_\theta}
\text{latent state}
\xrightarrow{g_\theta}
(\text{reward},\text{next latent state})
\xrightarrow{f_\theta}
(\text{policy},\text{value}).
$$

The planning loop is:

$$
\text{representation}
\rightarrow
\text{latent dynamics}
\rightarrow
\text{MCTS}
\rightarrow
\text{self-play target}
\rightarrow
\text{network update}.
$$

MuZero is therefore an important boundary case between:

- [[papers/architectures/world-models|World Models]], which emphasizes latent observation dynamics and a controller;
- [[papers/architectures/dream-to-control-learning-behaviors-by-latent-imagination|Dreamer]], which differentiates actor/value learning through imagined latent trajectories;
- [[papers/architectures/mastering-chess-and-shogi-by-self-play|AlphaZero]], which uses explicit search with known game dynamics;
- model-free value learning such as [[papers/architectures/human-level-control-through-deep-reinforcement-learning|DQN]].

## Problem setup

At time $t$, an agent receives an observation $o_t$, chooses an action $a_t$, and receives a reward $r_t$. The environment may be a board game with a known or unknown transition rule, or a visually observed Atari game whose true internal state is not directly available.

The agent seeks a policy $\pi$ that maximizes discounted return:

$$
J(\pi)
=
\mathbb{E}_{\pi,p}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right].
$$

The planning challenge is that a search procedure needs to evaluate hypothetical actions. In AlphaZero, the game rules provide the hypothetical next states. In MuZero, the learned dynamics function provides a latent successor state and the reward associated with the action.

The model does not need to predict every detail of the next observation. It needs to preserve enough information for the policy, value, and reward predictions used by search.

## Three network functions

MuZero separates the model into three conceptual functions.

| Function | Input | Output | Role |
| --- | --- | --- | --- |
| Representation $h_\theta$ | current observation or observation history | initial latent state $s^0$ | compresses what the agent has seen |
| Dynamics $g_\theta$ | latent state $s^k$, action $a^{k+1}$ | reward $\hat r^{k+1}$ and next latent state $s^{k+1}$ | advances the model during search |
| Prediction $f_\theta$ | latent state $s^k$ | policy prior $\hat p^k$ and value $\hat v^k$ | guides expansion and evaluates leaves |

The functions can be written as:

$$
s^0
=
h_\theta(o_{1:t}),
$$

$$
(r^{k+1},s^{k+1})
=
g_\theta(s^k,a^{k+1}),
$$

and

$$
(p^k,v^k)
=
f_\theta(s^k).
$$

The superscript $k$ denotes a hypothetical search depth, not necessarily a real environment time index. This distinction matters: after the root, the model operates on its own latent states rather than on newly observed pixels.

## Representation function

The representation function maps the available observation history to the root state:

$$
s^0=h_\theta(o_{1:t}).
$$

For a board game, the input may be a structured board representation. For Atari, it can be a stack of image frames or another observation encoding. The representation function is allowed to discard information that is not useful for the downstream planning targets.

This creates a deliberate information bottleneck:

$$
o_{1:t}
\rightarrow
s^0
\rightarrow
\{\text{reward, policy, value}\}.
$$

Unlike a conventional autoencoder, the architecture does not require a decoder that reconstructs $o_t$. The latent state is judged by the planning quantities it supports.

This is an important design lesson:

> A useful model for control need not be a faithful model of every observable detail.

The statement does not mean that observation prediction is never useful. It means that reconstruction is not a necessary architectural contract when the downstream decision process only needs task-relevant predictions.

## Dynamics function

Given a latent state and an action, the dynamics function produces a reward prediction and a successor latent state:

$$
(\hat r^{k+1},s^{k+1})
=
g_\theta(s^k,a^{k+1}).
$$

Repeated application creates a hypothetical trajectory:

$$
s^0
\xrightarrow{a^1}
s^1
\xrightarrow{a^2}
s^2
\xrightarrow{a^3}
\cdots.
$$

At each step, the model predicts the reward associated with the action:

$$
\hat r^{k+1}
=
r_\theta(s^k,a^{k+1}).
$$

The latent state is not required to be a decoded physical state. Two different latent states can represent different histories if they lead to different reward, policy, or value predictions, even when their observations look similar.

This is why “MuZero learns the environment” needs qualification. It learns a planning model, not necessarily a generally interpretable simulator.

## Prediction function

The prediction function maps a latent state to a policy prior and value estimate:

$$
(\hat p^k,\hat v^k)
=
f_\theta(s^k).
$$

The policy prior assigns probability to actions:

$$
\hat p^k(a)
=
\frac{\exp(\ell_a(s^k))}
{\sum_{b\in\mathcal{A}(s^k)}
\exp(\ell_b(s^k))}.
$$

The value head estimates expected future outcome:

$$
\hat v^k
\approx
\mathbb{E}
\left[
\sum_{j=0}^{\infty}
\gamma^j r_{k+j}
\mid s^k
\right].
$$

The policy prior does not have to be the final action distribution. It guides MCTS toward promising branches, while the value head evaluates leaves and backups.

## MCTS with a learned model

MuZero uses a tree search similar in spirit to AlphaZero. Each search edge stores a visit count $N(s,a)$, an accumulated value $W(s,a)$, and a prior probability $P(s,a)$. The mean action value is:

$$
Q(s,a)
=
\frac{W(s,a)}{N(s,a)}
$$

when $N(s,a)>0$.

A PUCT-style selection score can be written as:

$$
U(s,a)
=
Q(s,a)
+
c_{\mathrm{puct}}P(s,a)
\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}.
$$

The search selects the action with maximum $Q+U$. When an unexpanded edge is reached, MuZero applies the learned dynamics function:

$$
(r',s')
=
g_\theta(s,a).
$$

The prediction function then evaluates the new state:

$$
(p',v')
=
f_\theta(s').
$$

The policy prior expands the tree, the dynamics function moves through the latent tree, and the value function supplies a leaf evaluation. The environment is not queried for each hypothetical branch.

## Search policy target

After a fixed search budget, root visit counts produce an improved action distribution:

$$
\pi(a\mid s)
=
\frac{N(s,a)^{1/\tau}}
{\sum_bN(s,b)^{1/\tau}},
$$

where $\tau$ controls the sharpness of the distribution.

The search distribution contains more information than the action finally selected. It records how the search budget was allocated across alternatives and becomes a training target for the prediction policy.

The architecture therefore has two policy-like objects:

| Object | Source | Function |
| --- | --- | --- |
| Network prior $\hat p$ | prediction function | proposes promising branches |
| Search policy $\pi$ | MCTS visit counts | improves the action distribution using lookahead |

Confusing these two hides the central policy-improvement loop.

## Self-play and training data

For each real environment step, the agent can store the observation history, action, reward, and search policy. A trajectory contains:

$$
\mathcal{D}
=
\{(o_{1:t},a_t,r_t,\pi_t,z_t)\}_{t=1}^{T},
$$

where $\pi_t$ is the MCTS-improved policy target and $z_t$ is a value target, such as an observed or bootstrapped return.

The network is trained not only at the root. Starting from a real observation, it is unrolled through a sequence of actions using the learned dynamics function:

$$
s^0=h_\theta(o_{1:t}),
$$

$$
(r^{k+1},s^{k+1})
=
g_\theta(s^k,a_{t+k+1}),
$$

$$
(p^k,v^k)
=
f_\theta(s^k).
$$

The unrolled predictions are compared against reward, policy, and value targets from the real trajectory.

## Unrolled loss

A schematic loss for an unroll of length $K$ is:

$$
\mathcal{L}(\theta)
=
\sum_{k=0}^{K}
\left[
\ell_r(\hat r^k,r^k)
+
\ell_v(\hat v^k,z^k)
+
\ell_p(\hat p^k,\pi^k)
\right]
+
\lambda\|\theta\|_2^2.
$$

Typical terms are:

$$
\ell_r(\hat r,r)
=
\operatorname{CE}(\hat r,r)
$$

for a discretized reward representation or an appropriate regression loss,

$$
\ell_v(\hat v,z)
=
\operatorname{CE}(\hat v,z)
$$

when value targets are represented by a categorical support, and

$$
\ell_p(\hat p,\pi)
=
-\sum_a\pi(a)\log\hat p(a)
$$

for policy distillation from search.

The exact target encoding and gradient handling are implementation details that must be recorded during reproduction. The architectural invariant is that the same network supplies a root representation, repeatedly advances latent states, and predicts the quantities needed by search.

## Why reward, policy, and value are enough for planning

Suppose the agent is only evaluated by future rewards and action selection. Then a model can be useful if it preserves the information needed to answer:

1. what immediate reward follows from this action?
2. which actions are promising from the resulting state?
3. what is the expected long-term value of that state?

MuZero trains directly toward these answers:

$$
\text{latent state}
\rightarrow
(\hat r,\hat p,\hat v).
$$

It does not need to reconstruct every observation dimension if those dimensions do not affect these quantities.

This is an example of **task-oriented model learning**. The model class is shaped by the planner's interface rather than by a requirement to reproduce all environment variables.

## AlphaZero versus MuZero

The relationship to AlphaZero is easiest to see by comparing the transition source.

| Property | AlphaZero | MuZero |
| --- | --- | --- |
| Root representation | board state encoder | observation-history representation |
| Hypothetical transition | known game rules | learned latent dynamics |
| Search evaluation | policy/value network | policy/value network |
| Reward model | supplied by environment rules | learned reward prediction |
| Observation reconstruction | not required | not required |
| Domain assumption | legal actions and simulator available | dynamics may be unknown |
| Main risk | search and network cost | model error and model exploitation |

MuZero retains the policy/value/search loop but replaces the known transition function with $g_\theta$. That change allows the same planning template to be applied to visually observed environments whose transition rules are not supplied.

## World Models versus MuZero

World Models and MuZero both learn latent dynamics, but their interfaces differ.

| Dimension | World Models | MuZero |
| --- | --- | --- |
| Latent training signal | visual reconstruction and next-latent prediction | reward, policy, and value targets |
| Learned dynamics | probabilistic MDN-RNN | recurrent latent transition used by search |
| Control mechanism | controller optimized in imagined model | MCTS over latent states |
| Pixel decoder | central to VAE representation | not required by the planning contract |
| Main model output | latent future and reward | reward, latent future, policy, value |
| Planning style | rollout-based controller optimization | explicit tree search |

The difference is not that one is “model-based” and the other is not. Both are model-based. The distinction is which downstream interface determines what the model must learn.

## Dreamer versus MuZero

Dreamer and MuZero both use latent imagination, but their policy improvement mechanisms differ.

| Dimension | Dreamer | MuZero |
| --- | --- | --- |
| Latent rollout | differentiable imagined trajectory | search tree of latent trajectories |
| Policy update | actor optimization through value gradients | search-improved policy distillation |
| Value use | critic supplies return target | value head evaluates tree leaves |
| Branching | sampled policy rollout | explicit action branching |
| Main trade-off | model bias versus gradient efficiency | model bias versus search budget |

This comparison is useful when deciding whether a new agent should spend compute on many differentiable rollouts or on explicit branching and search.

## What the experiments establish

The paper evaluates MuZero on 57 Atari games and on Go, chess, and shogi. The reported results show that a learned planning model can achieve strong performance in visually complex Atari environments and match AlphaZero-level performance in the board-game settings tested, without being supplied the underlying game rules.

The strongest architectural conclusion is:

> A model that predicts planning-relevant quantities in a learned latent space can support tree search across environments with different observation and action interfaces.

The results do not prove that the latent state is a faithful simulator, that the same search budget is optimal across domains, or that model-based planning will always outperform a model-free baseline under equal compute.

## Ablation questions

- How much performance comes from MCTS versus the learned representation and prediction network?
- What happens when the dynamics model is used without search?
- How does search performance change as the unroll depth used for training changes?
- Does adding observation reconstruction improve or hurt planning-relevant prediction?
- How much does the model exploit reward-prediction errors in environments with delayed rewards?
- How do action branching factor and search budget affect the value of the learned model?
- Does the latent state preserve enough information for partial observability and history dependence?
- Are Atari and board-game gains driven by the same architecture components?
- What is the fair comparison when model inference, tree search, and environment interaction have different costs?

These questions separate planning, representation, model accuracy, and compute rather than treating MuZero as a single indivisible algorithm.

## Failure modes

### Model exploitation

Search can prefer a branch whose predicted reward is high because the dynamics model is wrong:

$$
\hat r_{\mathrm{model}}
\gg
r_{\mathrm{real}}.
$$

Increasing the search budget can make this worse if the search finds more opportunities to exploit a model error.

### Latent aliasing

If two histories map to states that look similar to the prediction head but have different future consequences, the model cannot plan reliably:

$$
h_\theta(o_{1:t})
\approx
h_\theta(o'_{1:t'})
\quad\text{while}\quad
p(\cdot\mid o_{1:t})
\ne
p(\cdot\mid o'_{1:t'}).
$$

The representation must preserve the history information relevant to future reward and action selection.

### Search cost

The model removes environment calls from hypothetical branches, but each simulation still requires repeated applications of $g_\theta$ and $f_\theta$. Search cost grows with the number of simulations, branching factor, and latent unroll depth.

### Target leakage and evaluation mismatch

Policy and value targets depend on the self-play procedure, search budget, replay policy, and evaluation protocol. A result can be incorrectly attributed to architecture if these parts are changed between baseline and proposed model.

## Complexity and scaling

Let $B$ be the number of MCTS simulations, $H$ the average search depth, and $C_g,C_f$ the costs of the dynamics and prediction functions. A rough per-decision model-search cost is:

$$
O\left(
B H (C_g+C_f)
\right).
$$

The environment interaction cost may be reduced, but model inference can dominate. Larger latent states may improve representation capacity while increasing memory traffic and search latency.

The total planning budget should therefore report:

| Budget | Why it matters |
| --- | --- |
| real environment steps | measures data efficiency |
| model updates | measures training compute |
| simulations per action | measures planning effort |
| latent unroll depth | measures model use per simulation |
| network evaluations | measures inference cost |
| replay size and sampling | determines target distribution |

## Reproduction checklist

- [ ] Specify the observation history and representation input format.
- [ ] Separate representation, dynamics, and prediction network parameters.
- [ ] Report whether reward, value, and policy targets use categorical support or regression.
- [ ] State the MCTS selection rule, exploration coefficient, simulation budget, and visit-count temperature.
- [ ] Record the latent unroll length and how gradients are handled through unrolled dynamics.
- [ ] Compare search with and without the learned model under matched network and compute budgets.
- [ ] Measure real-environment return separately from model-predicted return.
- [ ] Report model inference, search, training, and environment interaction costs.
- [ ] Evaluate multiple seeds and include low-budget and high-budget planning settings.
- [ ] Test whether observation reconstruction changes planning performance rather than assuming it does.

## Takeaway

MuZero turns the learned model into a planning interface rather than a full simulator. The architecture has three reusable contracts:

$$
\boxed{
h_\theta:\text{observations}\rightarrow\text{latent state}
}
$$

$$
\boxed{
g_\theta:\text{latent state}+\text{action}
\rightarrow
\text{reward}+\text{next latent state}
}
$$

$$
\boxed{
f_\theta:\text{latent state}
\rightarrow
\text{policy}+\text{value}
}
$$

MCTS composes those contracts into a policy-improvement loop. Its durable lesson is that a model should be trained for the decisions a planner must make, while its main risk is that search amplifies errors in precisely those learned predictions.

## Related notes

- [[ai/architectures|Architectures]]
- [[papers/architectures/mastering-chess-and-shogi-by-self-play|AlphaZero]]
- [[papers/architectures/world-models|World Models]]
- [[papers/architectures/dream-to-control-learning-behaviors-by-latent-imagination|Dreamer]]
- [[papers/architectures/human-level-control-through-deep-reinforcement-learning|DQN]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[agents/index|Agents]]
