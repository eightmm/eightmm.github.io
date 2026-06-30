---
title: State-Space Models
tags:
  - architectures
  - state-space-model
  - sequence-modeling
---

# State-Space Models

State-space models are sequence architectures that model long-range dependencies through recurrent state updates or structured sequence transformations.

A simple continuous state-space form is:

$$
\frac{dh(t)}{dt} = Ah(t) + Bx(t),
\qquad
y(t) = Ch(t)
$$

Sequence models use discretized or structured versions of this update to mix information over long contexts.

A discrete-time form is:

$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t,
\qquad
y_t = Ch_t + Dx_t
$$

The matrices may be constrained or parameterized so the sequence can be scanned efficiently.

## Key Ideas

- A state summarizes past sequence information and is updated as new inputs arrive.
- Structured parameterization can make long sequence mixing efficient compared with full [[concepts/architectures/attention|attention]].
- Some implementations behave like recurrent scans; others use parallel sequence transforms during training.
- State size, discretization, gating, and input-dependent parameters define what information can be retained.
- SSMs are often compared with [[concepts/architectures/rnn|RNNs]] and [[concepts/architectures/transformer|Transformers]] because all three solve sequence mixing differently.
- [[concepts/architectures/mamba|Mamba]] is best treated as a selective SSM rather than a separate top-level architecture family.

## Discretization Contract

A continuous system:

$$
\dot{h}(t)=Ah(t)+Bx(t)
$$

must be converted into a sequence update. A common form is:

$$
\bar{A}=\exp(\Delta A),
\qquad
\bar{B}=\left(\Delta A\right)^{-1}\left(\exp(\Delta A)-I\right)\Delta B
$$

where $\Delta$ is a step size or learned timescale. Papers should state how discretization, stability, and parameterization are handled.

## Convolution View

For linear time-invariant SSMs, the output can be written as a sequence convolution:

$$
y_t
=
\sum_{k=0}^{t} K_k x_{t-k}
$$

with kernel terms derived from powers of $\bar{A}$. This is why some SSMs can train with parallel convolution-like computation while still offering recurrent scanning for inference.

## Comparison Contract

| Family | Sequence Mixing | Strength | Risk |
| --- | --- | --- | --- |
| RNN/GRU/LSTM | recurrent hidden update | streaming and compact state | hidden bottleneck, sequential training |
| Transformer | pairwise attention | direct token-token interaction | $O(L^2)$ attention cost |
| Long convolution | implicit or explicit convolution kernel | long context without pairwise attention matrix | filter/gating design and kernel implementation matter |
| SSM | structured state or convolution kernel | long sequence efficiency | parameterization and stability details matter |
| Selective SSM | input-dependent state update | content-aware recurrence | harder to compare with fixed linear SSMs |

## Protein and Long Sequence Use

SSMs are attractive for long biological sequences:

$$
s=(a_1,\ldots,a_L), \qquad L \gg 10^3
$$

but a sequence model alone does not prove structure or function. For protein claims, check transfer tasks, family split, sequence identity, and whether structural context is included.

## Paper Evidence Boundary

| Claim | Required Evidence |
| --- | --- |
| longer context | evaluation at long $L$, not only theoretical scaling |
| faster than attention | wall-clock, memory, hardware, batch, and kernel details |
| better protein model | downstream task, split, and baseline protocol |
| stable training | initialization, discretization, normalization, sequence length |
| streaming inference | recurrent state cache and reset/chunk policy |

## Practical Checks

- Check whether the model is causal, bidirectional, or wrapped with extra context mixing.
- Track the effective context length claimed by the implementation, not just the theoretical state.
- Inspect how masks, padding, and variable-length sequences are handled.
- For protein tasks, verify whether residue order alone is modeled or whether structure, contacts, or family splits are also used.
- Are speed and memory claims measured on the same sequence length, hardware, and batch size?
- Is Mamba or another selective variant separated from linear SSM assumptions?

## Canonical Papers

| Paper | Why It Matters |
| --- | --- |
| [S4](/papers/architectures/s4) | structured state-space kernel for practical long-range sequence modeling |
| [Hyena](/papers/architectures/hyena) | gated implicit long convolution as a dense-attention-free sequence mixer |
| [Mamba](/papers/architectures/mamba) | selective state-space scan with input-dependent parameters |

## Related

- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[entities/protein|Protein]]
