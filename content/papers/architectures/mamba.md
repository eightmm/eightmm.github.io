---
title: Mamba
aliases:
  - papers/mamba
  - papers/selective-state-space-models
tags:
  - papers
  - architectures
  - state-space-model
  - sequence-modeling
---

# Mamba

> The paper introduced selective state-space models as a sequence-modeling backbone with input-dependent dynamics and linear-time scaling.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Mamba: Linear-Time Sequence Modeling with Selective State Spaces |
| Authors | Albert Gu, Tri Dao |
| Year | 2023 preprint; 2024 conference |
| Venue | COLM 2024 |
| arXiv | [2312.00752](https://arxiv.org/abs/2312.00752) |
| OpenReview | [tEYskw1VY2](https://openreview.net/forum?id=tEYskw1VY2) |
| Status | verified |

## Question

Transformers mix tokens with attention, but dense attention scales quadratically in sequence length. Previous subquadratic sequence models often struggled on information-dense discrete modalities. The question was whether a state-space sequence model could add content-dependent selection while keeping linear-time scaling.

The deeper architecture question is whether sequence models need explicit pairwise token-token attention to be expressive. Mamba argues that a recurrent state-space model can become competitive if its dynamics are selective: the model should decide, from the current input, what to remember, what to ignore, and what to expose.

This puts Mamba between several older ideas:

| Family | Strength | Weakness Mamba Targets |
| --- | --- | --- |
| RNNs | linear recurrence over sequence length | hard to scale and train as modern backbones |
| SSMs | long-range linear dynamics and efficient convolution/scan forms | limited content-dependent routing |
| Transformers | flexible content-based token mixing | quadratic attention cost |

## Main Claim

Selective state-space models make SSM parameters input-dependent, allowing sequence models to choose what to propagate or forget while retaining efficient long-sequence computation.

Narrowed claim:

$$
h_t
=
\bar{A}(x_t)h_{t-1}
+
\bar{B}(x_t)x_t
$$

$$
y_t
=
C(x_t)h_t
$$

where the transition and readout depend on the current input token.

The useful narrowed claim is:

$$
\text{input-dependent SSM}
+
\text{hardware-aware scan}
\Rightarrow
\text{linear-time sequence backbone competitive with attention baselines}
$$

This is not a claim that attention is obsolete. It is a claim that selective recurrence can be a serious backbone primitive.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | sequence of token or feature vectors |
| Output | sequence representations or next-token predictions |
| Main mixing primitive | selective state-space recurrence |
| Sequence scaling | linear in sequence length for the recurrent scan |
| Memory representation | compressed hidden state, not explicit all-pair attention |
| Selection mechanism | input-dependent SSM parameters and gates |
| Practical requirement | hardware-aware parallel scan/kernel implementation |

For a sequence:

$$
x_{1:T}
\in
\mathbb{R}^{T \times d}
$$

Mamba maps:

$$
x_{1:T}
\rightarrow
y_{1:T}
$$

through a recurrent state:

$$
h_t
=
f_\theta(h_{t-1}, x_t)
$$

The difference from a vanilla RNN is that $f_\theta$ has structured state-space dynamics and is implemented with a scan that can be parallelized efficiently.

## State-Space Starting Point

A continuous-time linear state-space model is:

$$
\frac{dh(t)}{dt}
=
Ah(t) + Bx(t)
$$

$$
y(t)
=
Ch(t)
$$

After discretization, a sequence model can be written as:

$$
h_t
=
\bar{A}h_{t-1}
+
\bar{B}x_t
$$

$$
y_t
=
Ch_t
$$

Classical SSMs use fixed parameters across the sequence. That makes them efficient and stable, but can make content-based selection difficult.

Mamba changes this by making key parameters depend on the current input:

$$
\bar{B}_t
=
\bar{B}(x_t)
$$

$$
C_t
=
C(x_t)
$$

and, in the broader selective formulation, the update can vary by token.

## Selectivity

Selectivity means the model can condition its memory update on the current token.

Conceptually:

$$
\text{token}
\rightarrow
\text{parameters controlling write, keep, and read}
$$

This is why Mamba is closer to gated recurrence than to a fixed linear filter. The model does not only run the same dynamics over every token; it can choose what information should enter the state and what should be emitted.

| Mechanism | Role |
| --- | --- |
| input-dependent $\Delta$ | controls discretization/time-step behavior |
| input-dependent $B$ | controls how input enters state |
| input-dependent $C$ | controls how state is read out |
| gating | controls the block output |

This is the paper's answer to the weakness of earlier SSMs on discrete, information-dense data.

## Method

Mamba combines:

| Component | Role |
| --- | --- |
| selective SSM | input-dependent recurrence over sequence length |
| hardware-aware scan | parallelizes recurrent computation efficiently |
| gated block design | builds a simple sequence backbone around the selective SSM |
| linear sequence scaling | avoids dense $T^2$ attention cost |

The architecture changes the sequence mixing primitive from dense pairwise attention to recurrent state updates with selective dynamics.

## Block Structure

A Mamba block can be read as a sequence mixing block with projection, local mixing, selective SSM, and gating.

| Stage | Role |
| --- | --- |
| input projection | expands or splits channels for SSM and gate paths |
| local convolution | adds short-range local context |
| selective SSM | performs recurrent long-range sequence mixing |
| gate path | modulates output |
| output projection | returns to model dimension |

The exact implementation details matter because Mamba's practical claim is not just mathematical recurrence. It depends on making the selective scan efficient on modern accelerators.

## Scan View

Although the recurrence is sequential in definition:

$$
h_t
=
\bar{A}_t h_{t-1}
+
\bar{B}_t x_t
$$

the computation can be implemented with a parallel scan under the right associative structure.

At a high level:

$$
\{(\bar{A}_t, \bar{B}_t x_t)\}_{t=1}^{T}
\xrightarrow{\text{scan}}
\{h_t\}_{t=1}^{T}
$$

This is the practical bridge between recurrence and high-throughput training.

## Complexity and Memory

| Model | Token Mixing | Sequence Cost | Memory Pattern |
| --- | --- | --- | --- |
| dense Transformer attention | pairwise token-token attention | $O(T^2)$ attention interactions | stores attention-like pair interactions |
| RNN | recurrent hidden state | $O(T)$ sequential recurrence | compressed state |
| Mamba | selective SSM scan | $O(T)$ sequence scaling | compressed state plus hardware-aware scan buffers |

The tradeoff is clear:

$$
\text{attention}
\Rightarrow
\text{explicit pairwise access}
$$

$$
\text{Mamba}
\Rightarrow
\text{compressed recurrent memory with selective updates}
$$

This can be much better for long sequences, but it changes what information is represented and how it is retrieved.

## Relation to Attention

Attention computes a content-dependent weighted sum over previous tokens:

$$
y_t
=
\sum_{j \le t}
\alpha_{tj} Vx_j
$$

Mamba instead updates a latent state:

$$
h_t
=
\bar{A}_t h_{t-1}
+
\bar{B}_t x_t
$$

and reads from it:

$$
y_t
=
C_t h_t
$$

So the question is not only efficiency. It is also about memory layout:

| Attention | Mamba |
| --- | --- |
| keeps access to many token states through pairwise weights | compresses history into recurrent state |
| flexible retrieval from context | efficient state update |
| quadratic dense form unless approximated | linear sequence scaling |
| easy to inspect attention weights, though not always faithful | harder to interpret hidden state memory |

## Relation to Mamba-2

[[papers/architectures/mamba-2|Mamba-2]] reframes the comparison between SSMs and attention through structured state space duality. The useful shift is from "attention vs recurrence" to "what structured sequence mixing matrix is being computed?"

Mamba can be expanded into a causal matrix mixer:

$$
y_t
=
\sum_{i\le t}
C_t
\left(
\prod_{j=i+1}^{t}A_j
\right)
B_i x_i.
$$

Mamba-2 studies this matrix view more directly and uses it to design a faster SSD/Mamba-style layer. For reading follow-up papers, separate:

| Axis | Mamba | Mamba-2 |
| --- | --- | --- |
| core contribution | selective SSM block | structured state space duality and SSD layer |
| main question | can input-dependent SSMs compete with attention? | how are SSMs and attention-like mixers related? |
| implementation angle | selective scan | faster block/chunk algorithms |

## Benchmark Reading

Mamba's benchmark claims should be read in three separate layers.

| Layer | What to check |
| --- | --- |
| quality | loss, accuracy, downstream task performance |
| scaling | sequence length, throughput, memory, wall-clock |
| implementation | whether custom kernels and hardware assumptions are used |

The paper is strongest when quality and efficiency are both considered. A linear-time architecture that loses too much quality is not enough; a competitive model that only wins with specific kernels should be described with that caveat.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Selective SSMs improve prior SSM weakness on discrete data | language modeling and synthetic/selective-copy style tasks | results depend on implementation and training setup |
| Mamba scales linearly with sequence length | algorithmic design and long-sequence experiments | hardware kernels are part of the practical claim |
| Mamba is competitive across modalities | language, audio, and genomics experiments | not a proof that attention is obsolete |

## Ablation Reading

Important ablation axes:

| Axis | What it tests | Reading |
| --- | --- | --- |
| selectivity vs fixed SSM | whether input-dependent parameters matter | central to the paper's contribution |
| scan/kernel implementation | whether theoretical scaling becomes practical speed | hardware-aware part of claim |
| block design | how much gating, convolution, and projection matter | Mamba is a block, not only an equation |
| sequence length | where linear scaling becomes visible | short-context tasks may not show the advantage |
| modality | language, audio, genomics, etc. | tests whether sequence bias transfers |

For this wiki, the main reusable idea is selectivity: make sequence memory update depend on the token.

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | sequence modeling |
| Input/output unit | sequence tokens to sequence representations or predictions |
| Main comparison | Transformer and prior subquadratic sequence models |
| Main metric family | language-modeling loss, downstream accuracy, throughput, sequence-length scaling |
| Not directly tested | all Transformer workloads, tool-use agents, structure-based molecular modeling |

## Implementation Notes

- The mathematical recurrence is simple to write but not enough for performance; scan kernels are part of the system.
- Long-sequence speedups depend on sequence length, batch size, hidden size, hardware, and kernel maturity.
- Mamba is not a drop-in proof that attention can be removed from every architecture.
- Hybrid attention-SSM designs may be better when exact retrieval from context matters.
- For biological sequences, the appealing part is long-context scaling, but benchmark design must avoid leakage and family overlap.
- For protein or genome modeling, the sequence length advantage does not remove the need for careful tokenization and evaluation splits.

## Limitations

- The paper's strongest practical claims depend on custom kernels and hardware-aware implementation.
- Linear-time scaling does not automatically imply better quality at every compute budget.
- Input-dependent recurrence changes interpretability and memory behavior relative to attention.
- Later Mamba variants may change the best block design, normalization, hybridization, and scaling behavior.
- Compressed recurrent state may be weaker than attention for tasks requiring precise retrieval of arbitrary earlier tokens.
- Linear sequence scaling does not automatically imply lower end-to-end latency in every setting.
- Comparisons against Transformers must control for data, training budget, implementation quality, and inference setup.

## Why It Matters

Mamba is a key anchor for modern state-space model architectures and alternatives to attention-based token mixing.

The reusable pattern is:

$$
\text{sequence}
\rightarrow
\text{input-dependent recurrent state update}
\rightarrow
\text{linear-time long-context backbone}
$$

This matters for architecture reading because it creates a third major branch:

| Branch | Anchor |
| --- | --- |
| attention sequence models | [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) |
| decoder-only LMs | [GPT-2](/papers/architectures/gpt-2) |
| selective state-space models | Mamba |

Mamba should be read as a serious alternative sequence primitive, especially when sequence length and throughput are first-order constraints.

## Connections

- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[papers/architectures/index|Architecture papers]]
