---
title: Universal Transformers
tags:
  - papers
  - architectures
  - transformer
  - recurrent
  - adaptive-computation
---

# Universal Transformers

> **One-line claim:** a Transformer can be made recurrent over computation depth, allowing every position to repeatedly refine its representation with shared self-attention and a dynamic halting mechanism.

## Citation

- Authors: Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser
- Venue: ICLR 2019
- Paper: [Universal Transformers](https://arxiv.org/abs/1807.03819)
- Version: arXiv v3, 2019

## Why this paper belongs in Architecture Papers

The standard Transformer applies a fixed stack of different layers. The Universal Transformer (UT) reuses one transition block across multiple computation steps:

$$
H^{(0)}
\xrightarrow{\text{shared transition}}
H^{(1)}
\xrightarrow{\text{shared transition}}
\cdots
\xrightarrow{\text{shared transition}}
H^{(t)}.
$$

The recurrence is over **depth**, not over input positions. All positions can still be updated in parallel at one step. This places the model between a feed-forward Transformer and an RNN:

| Model | Recurrence over tokens | Recurrence over computation | Parameters across steps |
| --- | --- | --- | --- |
| RNN | sequential positions | implicit | usually shared in time |
| Transformer | no | fixed depth | different blocks |
| Universal Transformer | no within a step | explicit depth recurrence | shared transition block |

This is an architectural paper because it changes the model's computation graph, parameter sharing, and effective depth. Its dynamic halting mechanism also makes computation an input-dependent quantity rather than a fixed hyperparameter.

## Problem setup

Let the input sequence contain $n$ positions and let each position have a $d$-dimensional state:

$$
H^{(t)}
\in
\mathbb{R}^{n\times d}.
$$

The initial state combines token embeddings and position information:

$$
H^{(0)}_i
=
E(x_i)+P_i.
$$

At computation step $t$, the same transition function updates all positions:

$$
H^{(t+1)}
=
\mathcal{T}_\theta(H^{(t)},t).
$$

The transition may include multi-head self-attention, a position-wise feed-forward network, residual connections, normalization, and a time-step representation. Unlike an ordinary Transformer stack, the parameters $\theta$ are reused at every recurrent step.

The output is read from the final state or from the state at which each position halts:

$$
\hat y_i
=
g_\omega(H_i^{(T_i)}),
$$

where $T_i$ may vary by position when adaptive computation is enabled.

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Token embedding | discrete or continuous input | token states | initializes content representation |
| Positional representation | position index and/or time step | positional signal | distinguishes input location and recurrent depth |
| Shared self-attention | all position states | globally mixed states | communicates between positions at one depth step |
| Shared feed-forward block | each position state | transformed state | mixes channels independently per position |
| Halting unit | current position state | halt probability | decides whether more computation is useful |
| Output head | final or halted state | task output | maps refined representations to predictions |

The contract separates two axes that are often conflated:

1. **sequence interaction:** which positions can communicate;
2. **computation allocation:** how many recurrent transitions each position receives.

Self-attention controls the first axis. Recurrence and halting control the second.

## Shared transition block

Ignoring the exact normalization order, one update can be written as:

$$
A^{(t)}
=
\operatorname{MHA}_\theta
\left(H^{(t)},H^{(t)},H^{(t)}\right),
$$

$$
U^{(t)}
=
\operatorname{Norm}
\left(H^{(t)}+A^{(t)}\right),
$$

$$
H^{(t+1)}
=
\operatorname{Norm}
\left(U^{(t)}+\operatorname{FFN}_\theta(U^{(t)})\right).
$$

The same $\operatorname{MHA}_\theta$ and $\operatorname{FFN}_\theta$ are applied at every $t$. The attention operation remains:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

The distinction from an ordinary Transformer is parameter sharing across transitions, not the removal of attention. A UT can therefore inherit the global receptive field of self-attention while acquiring an iterative refinement bias.

## Time-step representation

If the same transition block is reused indefinitely, the model needs a way to distinguish recurrent computation steps. A step representation $R_t$ can be added to the state:

$$
\widetilde{H}^{(t)}
=
H^{(t)}+R_t.
$$

This gives the shared block a notion of computation time. Without it, the transition is time-homogeneous:

$$
H^{(t+1)}=\mathcal{T}_\theta(H^{(t)}),
$$

which may still be useful, but it removes an explicit signal for early versus late refinement.

The time representation should not be confused with input position encoding. Position answers “where is this token in the sequence?”; time-step encoding answers “which recurrent refinement step is being executed?”

## Adaptive computation time

The model can attach a halting probability to each position. Let:

$$
p_i^{(t)}
=
\sigma
\left(
W_h H_i^{(t)}+b_h
\right)
$$

be the probability that position $i$ halts at step $t$. The accumulated halting mass is:

$$
S_i^{(t)}
=
\sum_{j=1}^{t}p_i^{(j)}.
$$

The position stops when the accumulated mass passes a threshold, subject to a maximum number of steps. The output can be a weighted mixture of intermediate states or the state at the halting step. A remainder term can preserve the final incomplete probability mass:

$$
r_i^{(t)}
=
1-S_i^{(t-1)}
$$

for the final transition, depending on the exact halting formulation.

The loss includes a ponder cost that discourages unnecessary computation:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{ponder}}
\sum_i T_i.
$$

The cost creates a quality-efficiency trade-off. With $\lambda_{\mathrm{ponder}}=0$, the model can use many steps to improve accuracy. A larger penalty encourages early halting but may stop before a difficult position has been resolved.

## Parallel in time, recurrent in depth

The phrase “parallel-in-time self-attentive recurrent” describes the main computation compromise. At a fixed recurrent step, all sequence positions are updated concurrently:

$$
\{H_i^{(t)}\}_{i=1}^{n}
\longrightarrow
\{H_i^{(t+1)}\}_{i=1}^{n}
$$

with one attention operation over the whole sequence. The model still needs multiple transitions, so total work scales with the number of steps:

$$
\text{cost}
\approx
T\cdot O(n^2d)
$$

for dense self-attention, ignoring projection and feed-forward terms. It is parallel over $n$ positions but not over the recurrent depth $T$.

This is different from an RNN that is sequential over positions and from a standard Transformer that is parallel over both positions and its fixed layer stack during training. The UT trades some depth parallelism for parameter sharing and iterative computation.

## Input-dependent depth

With adaptive halting, each position can use a different number of transitions:

$$
T_i
\in
\{1,\ldots,T_{\max}\}.
$$

This is useful when some tokens or subproblems are easy and others require more reasoning. However, position-wise halting does not automatically mean the hardware executes each position at a different physical time. Implementations may still pad to a common maximum or group positions by halting step. The algorithmic contract and the realized kernel behavior must therefore be measured separately.

## What the experiments establish

The paper evaluates Universal Transformers on algorithmic and language understanding tasks, including sequence copying, logical or relational tasks, language modeling, and machine translation. The reported evidence supports the following narrower claims:

- recurrent refinement can improve generalization on tasks where fixed-depth feed-forward processing struggles with length or algorithmic structure;
- adaptive computation can improve accuracy by allowing additional computation where needed;
- the architecture can retain parallel position updates while adding a recurrent computation axis;
- on the reported WMT14 English-German setup, the authors report a BLEU improvement over the Transformer baseline.

The result does not show that recurrent depth is universally better. It shows that the bias is useful on selected tasks and that the benefit should be examined alongside extra computation, halting policy, and training budget.

## Ablation questions

- How much of the gain comes from shared weights versus simply increasing the number of Transformer layers?
- Does the model still improve when the recurrent step count is fixed and adaptive halting is removed?
- Is the gain on length generalization preserved when training and evaluation use matched compute?
- Do positions actually halt at different depths, or does most computation still reach the maximum step?
- How sensitive is performance to the ponder cost and maximum number of transitions?
- Does adding a time-step representation matter when the same transition is reused?
- Are improvements consistent across algorithmic tasks, language modeling, and translation?

These comparisons are necessary because a UT can gain capacity, depth, and extra inference steps simultaneously.

## Complexity and memory

For sequence length $n$, hidden width $d$, and recurrent steps $T$, dense self-attention contributes approximately:

$$
O(Tn^2d)
$$

compute and:

$$
O(n^2)
$$

attention-score memory per active step, depending on whether attention matrices are materialized. The feed-forward part contributes approximately:

$$
O(Tndd_{\mathrm{ff}}).
$$

Parameter count does not grow linearly with $T$ because the transition block is shared. This makes deeper computation relatively cheap in parameters, but not necessarily cheap in latency or activation memory.

Adaptive halting can reduce average compute if many positions stop early. The reduction is real only when the implementation avoids doing the work for halted positions. A reported halting distribution should therefore be accompanied by wall-clock latency, token throughput, and maximum-step information.

## Relation to nearby architectures

| Paper or concept | Main difference |
| --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | fixed stack of encoder/decoder blocks; no recurrent depth or adaptive halting |
| [Long Short-Term Memory](/papers/architectures/long-short-term-memory) | recurrent state advances across sequence positions rather than applying shared attention across all positions at each depth step |
| [Neural Ordinary Differential Equations](/papers/architectures/neural-ode) | treats depth as continuous dynamics and uses an ODE solver instead of discrete shared Transformer transitions |
| [Mamba](/papers/architectures/mamba) | uses selective state-space recurrence along the sequence axis, not self-attention recurrence over computation depth |
| [Universal Transformers concept](/concepts/architectures/universal-transformer) | reusable wiki definition of shared depth recurrence and adaptive computation |
| [Adaptive computation](/concepts/architectures/adaptive-computation) | focuses on the computation-allocation mechanism independently of the Transformer backbone |

## Limits and failure modes

- Recurrent depth increases inference latency even when positions are parallelized.
- Weight sharing can limit the specialization that different Transformer layers provide.
- Halting probabilities are an optimization signal, not a proof that the model has found the minimum sufficient computation.
- Ponder penalties may trade accuracy for speed in a task-dependent way.
- Per-position halting is difficult to realize efficiently on dense hardware.
- Comparisons against fixed-depth baselines must match parameter count, training steps, and inference compute.
- Claims about algorithmic generalization are sensitive to length distribution, curriculum, and exact task generation.

## Implementation checklist

- [ ] Define whether recurrence is over depth, sequence position, or both.
- [ ] Decide which parameters are shared across recurrent steps.
- [ ] Add and document a computation-step representation if used.
- [ ] Log per-position halting steps, average steps, and maximum steps.
- [ ] Compare fixed-depth, shared-depth, and adaptive-halting variants.
- [ ] Match baselines by parameter count and total training/inference compute.
- [ ] Measure realized wall-clock speed rather than inferring efficiency from parameter count.
- [ ] Test length extrapolation with an explicit train/test length split.

## Takeaway

Universal Transformers make computation depth an explicit modeling axis. The architecture keeps self-attention's global communication and parallel updates, while adding recurrent refinement and optional input-dependent halting. It is a useful bridge between the fixed-depth Transformer, recurrent networks, and adaptive-computation systems.

## Related notes

- [[ai/architectures|Architectures]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/adaptive-computation|Adaptive computation]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/neural-ode|Neural Ordinary Differential Equations]]
