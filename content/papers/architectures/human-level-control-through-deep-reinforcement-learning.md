---
title: Human-level control through deep reinforcement learning
tags:
  - papers
  - architectures
  - reinforcement-learning
  - agents
---

# Human-level control through deep reinforcement learning

> **One-line claim:** a convolutional neural network can approximate an action-value function directly from pixels when experience replay and a separate target network make temporal-difference learning more stable.

## Citation

- Authors: Volodymyr Mnih et al.
- Venue: *Nature*, 518, 529-533 (2015)
- Paper: [Nature article](https://www.nature.com/articles/nature14236)
- DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)

## Why this paper belongs in Architecture Papers

The paper is usually introduced as a reinforcement-learning milestone, but its reusable contribution is an **agent architecture contract**:

$$
\text{observation frames}
\xrightarrow{\text{CNN}}
\text{state representation}
\xrightarrow{\text{action-value head}}
\{Q(s,a):a\in\mathcal{A}\}.
$$

The network does not emit a probability distribution over actions. It emits one scalar value per discrete action. The behavior policy then chooses an action from those values. This makes the boundary between representation learning, value estimation, and action selection explicit.

It is therefore different from a generic CNN paper and different from a policy-gradient paper. The CNN is the state encoder; the output head represents a value function; replay and target networks are training-system components that stabilize this architecture.

## Problem setup

At time $t$, the environment provides an observation $o_t$, the agent chooses $a_t$, receives reward $r_t$, and transitions to $o_{t+1}$. The paper uses preprocessed game frames and a short history of frames as the state:

$$
s_t = (x_{t-k+1},\ldots,x_t),
$$

where each $x_i$ is a processed image frame and $k$ is the number of frames in the input stack. The stack supplies limited temporal information without adding a recurrent hidden state.

The action space is discrete. The network therefore has one output for each legal action:

$$
Q_\theta(s_t,\cdot)\in\mathbb{R}^{|\mathcal{A}|}.
$$

The goal is not to reconstruct the image or predict the next frame. It is to estimate the expected discounted return obtained after taking an action:

$$
Q^\pi(s_t,a_t)
=
\mathbb{E}_\pi\left[
\sum_{j=0}^{\infty}\gamma^j r_{t+j}
\mid s_t,a_t
\right].
$$

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Frame preprocessing | raw game observation | image tensor | reduces visual input to a fixed representation |
| CNN encoder | stacked frames | latent feature vector | learns spatial and motion-relevant features |
| Fully connected value head | feature vector | $|\mathcal{A}|$ scalars | estimates one action value per discrete action |
| Behavior rule | action values | selected action | balances exploitation and exploration |
| Replay buffer | transitions | minibatch | breaks temporal correlation during optimization |
| Target network | next state | bootstrap values | supplies a slowly changing regression target |

The boundary is worth preserving in later designs. A new visual encoder can replace the CNN while the value head remains; a new action parameterization may require replacing the output head; replay and target updates are not architectural layers but affect whether the learned value function is trainable.

## The Q-learning target

The one-step Bellman target uses a frozen or slowly updated copy of the network, parameterized by $\theta^-$:

$$
y_t
=
r_t
+
\gamma(1-d_t)
\max_{a'} Q_{\theta^-}(s_{t+1},a'),
$$

where $d_t$ indicates a terminal transition. The online network is trained to match this target at the action that was actually taken:

$$
L(\theta)
=
\mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim\mathcal{D}}
\left[
\left(y_t-Q_\theta(s_t,a_t)\right)^2
\right].
$$

The target network is updated periodically from the online network:

$$
\theta^- \leftarrow \theta
\quad\text{every }C\text{ optimization steps}.
$$

This creates a useful separation of time scales. The online network changes every update, while the target used for bootstrapping changes less frequently. Without that separation, the model would chase a target that is itself moving with every gradient step.

## Experience replay

The agent stores transitions in a finite replay memory:

$$
\mathcal{D}
=
\{(s_t,a_t,r_t,s_{t+1})\}.
$$

At each training step, a minibatch is sampled approximately uniformly from $\mathcal{D}$. This has two architecture-level consequences:

1. The learner is decoupled from the exact order in which the environment generated samples.
2. The same transition can contribute to several updates while it remains in the buffer.

Replay is not a magical data augmentation method. It changes the sampling distribution seen by the optimizer and creates a lag between data collection and parameter updates. Its benefits depend on buffer size, transition diversity, reward scale, and the non-stationarity introduced by the changing behavior policy.

## Action selection

The paper uses an epsilon-greedy behavior policy. With probability $1-\epsilon$, the agent takes the action with the largest estimated value; otherwise it explores:

$$
a_t=
\begin{cases}
\arg\max_a Q_\theta(s_t,a), & \text{with probability }1-\epsilon,\\
\text{a random action}, & \text{with probability }\epsilon.
\end{cases}
$$

The behavior policy is not the same object as the Q-network. The network estimates values; epsilon-greedy converts those estimates into behavior. This distinction matters when comparing DQN with policy networks, actor-critic systems, or an LLM agent that selects tools directly.

## Visual encoder

The input is a fixed stack of processed frames. The convolutional stack progressively changes the representation from local image patterns to a compact feature vector. The final fully connected layers map that vector to action values.

The important architectural choice is not the exact historical kernel table by itself. It is the decision to share one visual representation across all action values:

$$
Q_\theta(s,a)=h_a\bigl(f_\theta(s)\bigr),
$$

where $f_\theta$ is the shared image encoder and $h_a$ is the action-specific value readout. This is cheaper and statistically more efficient than learning a separate image encoder for every action.

The shared representation also creates a failure mode: if a visual distinction is useful for one action but discarded by the encoder, the value head cannot recover it. Representation quality and action-value accuracy are therefore coupled.

## What the experiments establish

The paper evaluates the same general agent design across 49 Atari 2600 games using only pixels, game score, and a common training procedure. Its evidence supports three narrower claims:

- a single value-based architecture can learn useful visual representations across many games;
- replay and a separate target network make deep Q-learning substantially more workable;
- the resulting agent can reach strong performance on a broad set of discrete-action tasks.

The result should not be generalized to “CNN plus Q-learning solves arbitrary control.” The benchmark has a fixed action interface, relatively short observations, and a reward signal supplied by the game. The architecture does not by itself solve long-horizon planning, partial observability, continuous actions, or safe exploration.

## Ablation questions

The Nature page identifies extended comparisons for replay and target-network separation. When reading the full paper and supplement, check:

- What happens when experience replay is removed?
- What happens when the target network is replaced by the online network?
- Is the gain from the CNN representation separable from the gain from the optimization protocol?
- Does one common architecture work equally well across all games, or are the aggregate results hiding large per-game variation?
- How sensitive are results to frame preprocessing, frame history, reward clipping, and exploration schedule?

These questions prevent the paper from being reduced to a single benchmark score.

## Complexity and scaling

For a CNN encoder with spatial feature maps, the convolutional cost is roughly proportional to the number of output positions, input channels, output channels, and kernel elements. The action-value head adds a cost proportional to $|\mathcal{A}|$ times the final feature width:

$$
\text{readout cost}
\approx
O\left(|\mathcal{A}|d\right),
$$

where $d$ is the encoder feature dimension. This is attractive for a small discrete action set. It becomes a poor fit when actions are continuous, combinatorial, or parameterized by objects in the observation.

Replay also increases memory use:

$$
O\left(|\mathcal{D}|\cdot\text{transition size}\right).
$$

In practical systems, storing raw frames can dominate memory, so frame compression, lazy stacking, and storage layout become part of the system design even though they are not neural layers.

## Relation to nearby architectures

| Paper or concept | Main difference |
| --- | --- |
| [AlexNet](/papers/architectures/alexnet) | visual feature extractor for classification rather than value estimation |
| [LSTM](/papers/architectures/long-short-term-memory) | adds recurrent memory when a frame stack is insufficient |
| [Reinforcement learning](/concepts/learning/reinforcement-learning) | defines the learning problem; DQN is one value-based architecture and training recipe |
| [Agents](/agents) | describes a broader observe-plan-act interface; DQN provides a learned action-value controller |
| [World Models](/papers/architectures/world-models) | learns a latent dynamics model and trains a controller through predicted rollouts |
| [Differentiable Neural Computer](/papers/architectures/differentiable-neural-computer) | external memory is explicit, while DQN's replay is an optimization buffer rather than a differentiable memory |

## Limits

- The action output is naturally suited to a discrete action set.
- A frame stack is only a finite approximation to memory and does not guarantee state observability.
- Bootstrapping can propagate erroneous value estimates and overestimation.
- Replay data comes from a changing policy, so the learning distribution is not fixed.
- Reward design and reward clipping can change the behavior that is learned.
- Atari performance does not establish transfer to a new visual domain or a new action interface.

## Implementation checklist

- [ ] Define the observation preprocessing and temporal context explicitly.
- [ ] Separate online and target network parameters.
- [ ] Record terminal transitions correctly in the bootstrap target.
- [ ] Sample replay transitions without accidentally correlating minibatches.
- [ ] Log action-value scale, reward scale, replay age, and exploration rate.
- [ ] Compare replay and target-network ablations under the same environment budget.
- [ ] Report per-task results rather than only an aggregate score.

## Takeaway

DQN is a clean example of an agent architecture whose pieces have different jobs: a CNN represents observations, a value head scores actions, a behavior rule chooses actions, and replay plus a target network make the value-learning loop less unstable. That separation is more reusable than the historical Atari configuration.

## Related notes

- [[ai/architectures|Architectures]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[agents/index|Agents]]
- [[papers/architectures/world-models|World Models]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
