---
title: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
tags:
  - papers
  - architectures
  - agents
  - reinforcement-learning
  - tree-search
---

# Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm

> **One-line claim:** a single policy/value network combined with Monte Carlo Tree Search can learn strong play from self-play using only the game rules, without human games or handcrafted evaluation features.

## Citation

- Authors: David Silver et al.
- Year: 2017
- Paper: [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- Common name: AlphaZero

## Why this paper belongs in Architecture Papers

AlphaZero is an **agent architecture**, not only a reinforcement-learning objective. Its behavior comes from the composition of three modules:

$$
\text{state encoder}
\xrightarrow{f_\theta}
(\text{policy prior},\text{value estimate})
\xrightarrow{\text{MCTS}}
\text{improved action distribution}
\xrightarrow{\text{action selection}}
\text{environment transition}.
$$

The neural network does not directly choose every final action. It supplies priors and leaf evaluations to a search procedure. The search then improves the policy used for self-play, and those improved decisions become training targets for the same network.

This creates a feedback loop between learned representation and explicit planning:

$$
\text{network}
\rightarrow
\text{search}
\rightarrow
\text{self-play data}
\rightarrow
\text{network update}.
$$

That contract is useful for understanding game agents, model-based planning, tool-selection agents, and systems that combine a learned proposal with explicit search.

## Problem setup

At state $s_t$, the agent chooses action $a_t$ from a legal action set $\mathcal{A}(s_t)$, transitions to $s_{t+1}$, and eventually receives a terminal outcome $z\in\{-1,0,+1\}$ from the perspective of the current player.

The network maps a state to two outputs:

$$
(p_\theta(\cdot\mid s),v_\theta(s))=f_\theta(s),
$$

where:

- $p_\theta(a\mid s)$ is a prior probability over legal actions;
- $v_\theta(s)$ estimates the expected game outcome from state $s$.

The policy head proposes promising branches. The value head evaluates positions that the search reaches. The two heads share a representation trunk.

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| State representation | board state and current-player information | spatial feature tensor | represents pieces, locations, and legal context |
| Shared trunk | state features | latent board representation | extracts reusable position features |
| Policy head | trunk representation | action prior $p_\theta$ | guides tree expansion and action sampling |
| Value head | trunk representation | scalar $v_\theta$ | estimates eventual outcome at a leaf |
| MCTS | root state, priors, values, rules | visit-count distribution $\pi$ | performs explicit lookahead and improves action selection |
| Self-play loop | search policy and game rules | trajectories | supplies training states, search policies, and outcomes |

The architecture is domain-general at the algorithmic level, but the state encoder and legal-action interface depend on the game. “No domain knowledge” therefore means no human strategy or handcrafted evaluation function beyond the rules and state/action representation required to operate the game.

## Policy and value network

The trunk can be expressed abstractly as:

$$
h=f_{\mathrm{trunk},\theta}(s).
$$

The policy head produces logits for actions:

$$
\ell_a=W_a h+b_a,
\qquad
p_\theta(a\mid s)
=
\frac{\exp(\ell_a)}
{\sum_{b\in\mathcal{A}(s)}\exp(\ell_b)}.
$$

The value head produces a bounded scalar:

$$
v_\theta(s)=\tanh(W_vh+b_v).
$$

The shared trunk allows policy and value to use related features while the heads specialize their readout. This is a multi-task architecture: action priors and outcome estimates are trained from the same self-play states.

## Monte Carlo Tree Search

For each search root, MCTS repeatedly performs four conceptual phases:

1. select a path using an exploration rule;
2. expand an unvisited node with the network policy prior;
3. evaluate the new leaf with the network value head;
4. backup the value through the selected path.

Let $N(s,a)$ be an edge visit count, $W(s,a)$ the accumulated value, and:

$$
Q(s,a)=\frac{W(s,a)}{N(s,a)}
$$

the mean action value. A PUCT-style selection score can be written as:

$$
U(s,a)
=
Q(s,a)
+
c_{\mathrm{puct}}
P(s,a)
\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}.
$$

The selected edge maximizes $Q+U$. The prior $P(s,a)$ comes from the policy head. The exploration term is large for actions with high prior and low visit count, and shrinks as an edge is explored.

After many simulations, the search policy is formed from root visit counts:

$$
\pi(a\mid s)
=
\frac{N(s,a)^{1/\tau}}
{\sum_bN(s,b)^{1/\tau}},
$$

where $\tau$ controls how sharply visits become an action distribution. The exact temperature schedule and root-noise procedure are part of the self-play protocol, not merely implementation details.

## Search as policy improvement

The raw network policy $p_\theta$ is not used as the final self-play policy. MCTS converts it into a stronger root distribution $\pi$ by allocating simulations to promising actions and using value estimates to compare future outcomes.

The relationship is:

$$
p_\theta(a\mid s)
\xrightarrow{\text{tree search}}
\pi(a\mid s).
$$

The improved distribution becomes a supervised target for the policy head. This is a form of policy iteration in which the learned network proposes and evaluates while search performs a local improvement step.

## Self-play data

A self-play game produces states, search distributions, and final outcomes:

$$
\mathcal{D}
=
\{(s_t,\pi_t,z_t)\}_{t=1}^{T}.
$$

The state target contains more information than the final action alone. Visit counts encode which alternatives search considered plausible and how the search budget was allocated.

The network loss combines value regression and policy imitation of search:

$$
\mathcal{L}(\theta)
=
\sum_t
\left[
\left(z_t-v_\theta(s_t)\right)^2
-
\pi_t^\top\log p_\theta(\cdot\mid s_t)
\right]
+
\lambda\|\theta\|_2^2.
$$

The first term trains outcome prediction. The second term trains the network to reproduce the search-improved action distribution. The regularizer is shown generically; a complete reproduction must use the paper's exact optimization and regularization settings.

## The iterative loop

The high-level training procedure alternates between data generation and parameter updates:

$$
\theta_k
\rightarrow
\text{self-play with MCTS}
\rightarrow
\mathcal{D}_k
\rightarrow
\theta_{k+1}.
$$

The updated network changes both the priors and values used by later searches. This makes the data distribution non-stationary. Replay windows, number of self-play games, search simulations, evaluation gating, and checkpoint selection all affect the resulting agent.

## What the experiments establish

The paper reports one algorithm applied to chess, shogi, and Go, starting from random play and using only the rules of each game. It reports superhuman-level performance in the tested domains and victories against strong existing programs.

The narrower architectural claim is that a shared policy/value network plus MCTS and self-play can support strong play across different board games with the game-specific state and legal-action interfaces changed. The result does not establish that the same design transfers automatically to stochastic, partially observed, continuous-action, or open-world environments.

## Ablation questions

- How much strength comes from the neural network versus the number of MCTS simulations?
- What happens when the value head is removed or replaced by a handcrafted evaluator?
- How sensitive is training to root exploration noise and temperature schedules?
- Does the policy head become better because of search targets, or simply because more self-play data is generated?
- How does performance change with fewer simulations per move or a smaller network?
- Are comparisons made with matched hardware, time, and search budgets?
- Does the game-specific state encoding leak useful domain assumptions despite the general algorithm claim?
- How stable are results across independent runs and checkpoint selection rules?

These questions separate architecture, planning budget, data generation, and evaluation opponent strength.

## Complexity and scaling

If each move uses $S$ simulations and each simulation traverses depth $D$, the search work is roughly:

$$
O(SD)
$$

tree operations, plus neural-network evaluations for expanded leaves. Batched leaf evaluation can improve hardware utilization, while deeper or wider search increases decision latency.

The total training cost includes self-play and network optimization:

$$
C_{\mathrm{total}}
=
C_{\mathrm{self\text{-}play}}
+
C_{\mathrm{training}}
+
C_{\mathrm{evaluation}}.
$$

An agent may have a small policy/value network but a large inference-time planning cost. Reporting only parameter count is therefore insufficient.

## Relation to nearby architectures and methods

| Paper or concept | Main difference |
| --- | --- |
| [DQN](/papers/architectures/human-level-control-through-deep-reinforcement-learning) | predicts action values and selects actions directly with an epsilon-greedy rule; AlphaZero adds explicit tree search and a policy head |
| [World Models](/papers/architectures/world-models) | learns latent environment dynamics for imagined rollouts; AlphaZero uses known game rules and search rather than a learned transition model |
| [Neural Architecture Search with Reinforcement Learning](/papers/architectures/neural-architecture-search-with-reinforcement-learning) | uses RL to generate model descriptions; AlphaZero uses search to choose environment actions |
| [Agents](/agents) | broader observe-plan-act system boundary; AlphaZero is a specialized planning agent architecture |
| [Universal Transformers](/papers/architectures/universal-transformers) | recurrent computation depth inside a neural model, not external tree search over actions |

## Limits and failure modes

- The environment must provide a usable rules engine and legal-action generator.
- Search cost can dominate both training and inference.
- Value estimates can be miscalibrated outside states reached by self-play.
- The policy/value heads share representation capacity and can create competing gradients.
- Evaluation against a fixed opponent can reward exploitation of that opponent rather than general strength.
- The architecture is less direct for continuous action spaces or environments with expensive branching factors.
- “No human data” does not mean “no design choices”; state encoding, action encoding, search constants, and training schedule remain important.

## Implementation checklist

- [ ] Define state, legal-action, terminal-result, and player-perspective conventions.
- [ ] Separate policy priors from value estimates in the network interface.
- [ ] Log visit counts, search simulations, root entropy, and selected actions.
- [ ] Verify value sign handling during backup from alternating player perspectives.
- [ ] Store search policy targets rather than only final actions.
- [ ] Match self-play, replay, search, and training budgets in comparisons.
- [ ] Evaluate against multiple opponents and independent checkpoints.
- [ ] Report wall-clock inference cost in addition to network FLOPs.

## Takeaway

AlphaZero is a canonical architecture for combining learned representation with explicit planning. The policy head proposes actions, the value head evaluates positions, MCTS allocates lookahead, and self-play turns search behavior into training data. Its most reusable lesson is the interface between a neural proposal/evaluator and an external decision procedure.

## Related notes

- [[ai/architectures|Architectures]]
- [[agents/index|Agents]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[papers/architectures/human-level-control-through-deep-reinforcement-learning|DQN]]
- [[papers/architectures/world-models|World Models]]
- [[papers/architectures/neural-architecture-search-with-reinforcement-learning|Neural Architecture Search with Reinforcement Learning]]
