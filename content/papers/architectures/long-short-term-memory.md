---
title: Long Short-Term Memory
aliases:
  - papers/lstm
  - papers/long-short-term-memory
tags:
  - papers
  - architectures
  - recurrent
---

# Long Short-Term Memory

> The paper introduced gated recurrent memory cells designed to preserve error signals over long time spans.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Long Short-Term Memory |
| Authors | Sepp Hochreiter, Jurgen Schmidhuber |
| Year | 1997 |
| Venue | Neural Computation |
| DOI | [10.1162/neco.1997.9.8.1735](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) |
| Status | verified |

## Question

The paper asks whether a recurrent neural network can learn dependencies over long time lags without losing the training signal through repeated nonlinear transitions.

A vanilla recurrent model updates:

$$
h_t = \phi(W_x x_t + W_h h_{t-1} + b)
$$

and predicts from $h_t$. When the loss at a late time step depends on an early input, the gradient has to pass through many Jacobian products:

$$
\frac{\partial \mathcal{L}}{\partial h_k}
=
\frac{\partial \mathcal{L}}{\partial h_t}
\prod_{j=k+1}^{t}
\frac{\partial h_j}{\partial h_{j-1}}.
$$

If the dominant singular values of these transition Jacobians are consistently below one, gradients shrink exponentially. If they are above one, gradients can explode. The architecture question is therefore not just "can recurrence represent memory?", but "can the architecture expose a trainable path along which error can remain stable?"

LSTM answers this by separating a memory cell from the exposed hidden state and controlling read/write behavior with gates.

## Main Claim

Long Short-Term Memory introduces a recurrent memory cell with multiplicative gates so that information and error signals can persist over long time intervals.

In modern terms, the key architectural claim is:

$$
\text{protected additive memory path}
+
\text{learned gates}
\Rightarrow
\text{better long-range sequence learning than plain RNN recurrence}.
$$

The paper is important because it makes memory an explicit architectural object. It is not simply a different activation function or optimizer trick. It changes the state update so that part of the recurrent computation can behave like a controlled accumulator.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | ordered sequence $(x_1,\ldots,x_T)$ |
| Output | hidden sequence $(h_1,\ldots,h_T)$ or task-specific prediction from hidden states |
| State | exposed hidden state $h_t$ plus internal cell state $c_t$ |
| Token mixing | causal recurrent update from $t-1$ to $t$ |
| Memory mechanism | additive cell state path with gated writes, retention, and reads |
| Parallelism | sequential over time; batch-parallel across independent sequences |
| Inductive bias | temporal order, causal state, compressive memory |

The model processes one step at a time:

$$
(h_t, c_t) = \operatorname{LSTMCell}(x_t, h_{t-1}, c_{t-1}).
$$

For a sequence classification task:

$$
\hat{y} = g(h_T)
$$

or, for token-level prediction:

$$
\hat{y}_t = g(h_t).
$$

The important constraint is that all information from earlier positions must be compressed into the recurrent state. This differs from [[concepts/architectures/attention|attention]], where later tokens can directly read earlier token representations.

## Core Equations

The commonly used LSTM equations are usually written as:

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

where:

| Symbol | Meaning |
| --- | --- |
| $x_t$ | input at time step $t$ |
| $h_t$ | exposed hidden state |
| $c_t$ | internal cell state |
| $i_t$ | input gate; controls write strength |
| $f_t$ | forget gate; controls retention of previous cell state |
| $o_t$ | output gate; controls how much memory is exposed |
| $\tilde{c}_t$ | candidate cell content |
| $\sigma$ | sigmoid gate in $[0,1]$ |
| $\odot$ | elementwise product |

The original 1997 presentation differs from the exact modern LSTM recipe; the forget gate became a standard addition through later variants. But the architectural idea is the same: protect a cell state and learn when to write, keep, and expose it.

## Why The Additive Cell Path Matters

The central path is:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t.
$$

The derivative from $c_t$ to $c_{t-1}$ is approximately:

$$
\frac{\partial c_t}{\partial c_{t-1}} \approx f_t
$$

ignoring indirect dependencies of gates on $h_{t-1}$. If a dimension of $f_t$ stays near $1$, that memory dimension can carry information and gradients across many steps:

$$
\frac{\partial c_t}{\partial c_k}
\approx
\prod_{j=k+1}^{t} f_j.
$$

This is still not magic. If the forget gates are usually far below one, memory decays. If they are near one for irrelevant information, stale state can persist. LSTM gives the model a learnable mechanism for memory retention; it does not remove the need to learn the right retention policy.

The important architecture pattern is additive state update plus multiplicative control. This pattern later reappears in different forms:

- [[concepts/architectures/gru|GRU]] uses fewer gates and merges parts of the memory/update logic.
- [[concepts/architectures/residual-connection|Residual connections]] provide additive identity paths across depth instead of time.
- [[concepts/architectures/state-space-model|State-space models]] define recurrent dynamics with structured transition rules.
- [[concepts/architectures/mamba|Mamba]] makes state-space transitions input-dependent.

## Gate Roles

| Gate | Equation | Role | Failure Mode |
| --- | --- | --- | --- |
| Input gate | $i_t = \sigma(\cdot)$ | decides how much new candidate content enters memory | under-writing misses useful evidence; over-writing destroys memory |
| Forget gate | $f_t = \sigma(\cdot)$ | decides how much previous memory remains | too low causes decay; too high keeps stale state |
| Output gate | $o_t = \sigma(\cdot)$ | decides how much memory becomes visible in $h_t$ | hidden state can expose too little or too much |
| Candidate | $\tilde{c}_t=\tanh(\cdot)$ | proposes new content | candidate bottleneck limits what can be stored |

The gates are multiplicative selectors. They are not independent modules that understand semantics by themselves. They are learned functions of the current input and previous hidden state:

$$
\begin{bmatrix}
i_t \\
f_t \\
o_t \\
\tilde{c}_t
\end{bmatrix}
=
\begin{bmatrix}
\sigma \\
\sigma \\
\sigma \\
\tanh
\end{bmatrix}
\left(
W x_t + U h_{t-1} + b
\right)
$$

where implementations often fuse the four affine projections for efficiency.

## Block View

| Block | Input | Output | Architectural Reason |
| --- | --- | --- | --- |
| Recurrent affine projections | $x_t, h_{t-1}$ | gate logits and candidate logits | compute input-conditioned control signals |
| Sigmoid gates | logits | $i_t, f_t, o_t$ | constrain control values to $[0,1]$ |
| Candidate transform | logits | $\tilde{c}_t$ | propose bounded write content |
| Cell update | $c_{t-1}, i_t, f_t, \tilde{c}_t$ | $c_t$ | additive memory update |
| Output transform | $c_t, o_t$ | $h_t$ | expose controlled memory to the next layer/task |

This decomposition is useful when reading later architectures. Many modern blocks can be read as variants of:

$$
\text{new state}
=
\text{keep gate} \cdot \text{old state}
+
\text{write gate} \cdot \text{new content}.
$$

## Relation To Vanilla RNNs

| Property | Vanilla RNN | LSTM |
| --- | --- | --- |
| State | one hidden state $h_t$ | hidden state $h_t$ plus cell state $c_t$ |
| Update | fully nonlinear transition | gated additive memory update |
| Long-range gradient | repeated Jacobian product through nonlinear transition | controlled path through cell state |
| Memory control | implicit in recurrent weights | explicit read/write/retention gates |
| Parallelism over time | sequential | sequential |
| Inductive bias | causal compressive recurrence | causal compressive recurrence with protected memory |

The LSTM does not change the sequential nature of recurrence. It changes the internal state geometry so that preserving information becomes easier.

## Relation To GRU

The [[concepts/architectures/gru|GRU]] can be read as a simpler gated recurrence:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}) \\
r_t &= \sigma(W_r x_t + U_r h_{t-1}) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h(r_t \odot h_{t-1})) \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t.
\end{aligned}
$$

GRU merges the cell and hidden state and uses update/reset gates. LSTM keeps a separate cell state and output gate. The practical choice depends on task scale, implementation, regularization, and historical codebase constraints rather than a universal dominance relation.

## Relation To Transformers

The [[papers/architectures/attention-is-all-you-need|Transformer]] changes the sequence modeling bottleneck.

| Dimension | LSTM | Transformer |
| --- | --- | --- |
| Token interaction | information flows through recurrent state | tokens attend directly to other tokens |
| Training parallelism | sequential over time | parallel over sequence positions under teacher forcing |
| Long context path length | $O(T)$ recurrent steps | one attention layer can connect any pair |
| Memory representation | compressed state vector | explicit sequence of token states |
| Main cost pattern | $O(Td^2)$ for recurrent projections | $O(T^2d)$ attention plus projections |

The Transformer became the default for large language models because direct token-token interaction and parallel training scale well on modern accelerators. LSTM remains useful as a reference for:

- streaming inference with bounded state;
- compact sequence models;
- causal filters with small memory;
- architecture discussions about gated state updates;
- historical understanding of why attention was disruptive.

## Relation To State-Space Models

State-space models and LSTMs both process sequences through recurrent state, but they organize the state update differently.

A simple linear state-space model has:

$$
\begin{aligned}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t.
\end{aligned}
$$

LSTM uses nonlinear, gated, input-dependent memory:

$$
c_t = f_t(x_t,h_{t-1}) \odot c_{t-1}
+ i_t(x_t,h_{t-1}) \odot \tilde{c}_t.
$$

Modern selective state-space models such as [[papers/architectures/mamba|Mamba]] revisit recurrent sequence modeling with hardware-aware scan algorithms and input-dependent parameters. LSTM is therefore not just an obsolete pre-Transformer artifact. It is a foundational example of controlled recurrence.

## Evidence Reading

The paper's evidence should be read in the context of the 1990s recurrent-learning problem. The reported tasks were small compared with modern benchmarks, but they directly targeted long time lags and gradient propagation.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| LSTM can bridge long time lags | synthetic delay tasks | gated memory can preserve information over many steps | synthetic tasks do not guarantee broad real-world performance |
| Constant error flow helps training | comparison against prior recurrent methods | architecture changes gradient behavior | later variants and optimizers changed practical recipes |
| Memory cells can learn to store and release information | task-specific sequence experiments | explicit gates can learn retention policies | small-scale experiments by modern standards |

The paper should not be read as proving that LSTM solves all long-context learning. It proves that changing the recurrent architecture can materially change trainability over time.

## What To Check In Later LSTM Papers

When reading later LSTM variants or applications, separate these factors:

| Factor | Question |
| --- | --- |
| Cell variant | Is it original LSTM, forget-gate LSTM, peephole LSTM, stacked LSTM, bidirectional LSTM, or projected LSTM? |
| Objective | Is the gain from architecture or from sequence-to-sequence training, CTC, language modeling, or another objective? |
| Context length | How long are the dependencies actually tested? |
| Parallelism cost | Is sequential recurrence acceptable for the target hardware and latency regime? |
| State size | Is the recurrent state large enough to carry required information? |
| Regularization | Are dropout, layer norm, clipping, or weight tying part of the reported gain? |

This matters because many historical sequence-model improvements bundle architecture, objective, decoding, data, and optimization changes.

## Implementation Notes

Most practical implementations fuse the gate projections:

$$
Z_t = W x_t + U h_{t-1} + b
$$

then split:

$$
Z_t = [z_i, z_f, z_o, z_c]
$$

and apply:

$$
i_t=\sigma(z_i),\quad
f_t=\sigma(z_f),\quad
o_t=\sigma(z_o),\quad
\tilde{c}_t=\tanh(z_c).
$$

Important engineering details:

- initialize forget-gate bias carefully when long retention is needed;
- use gradient clipping for unstable sequence training;
- mask padded positions so hidden states do not learn from padding tokens;
- decide whether hidden state should reset between sequences, documents, or batches;
- use packed sequences or length-aware batching for variable-length data;
- avoid comparing LSTM and Transformer models without controlling parameter count, training tokens, and compute.

## Common Misreadings

### "LSTM removes vanishing gradients"

It reduces a major failure mode by adding controlled memory paths. It does not guarantee stable gradients for every sequence, task, initialization, or optimizer.

### "The cell state is an external memory"

The cell state is internal recurrent state. It is not an addressable memory table like attention over stored tokens.

### "Forget gates were in the original form exactly as used today"

The modern equations usually include forget gates and other later conventions. The original paper's core idea is gated memory and error preservation, but the common implementation lineage includes subsequent refinements.

### "Transformers made LSTM irrelevant"

Transformers changed the default for large-scale sequence modeling. LSTM remains relevant for compact recurrent baselines, streaming systems, historical comparison, and any architecture discussion about state, gating, and memory.

## Why It Still Matters

LSTM is the cleanest canonical architecture for understanding:

- [[concepts/architectures/rnn|recurrent neural networks]];
- [[concepts/architectures/gating|gating]];
- vanishing and exploding gradients through time;
- compressive sequence memory;
- the pre-Transformer sequence modeling baseline;
- why direct attention and state-space models are architectural alternatives, not just implementation tricks.

For an AI architecture wiki, LSTM should remain close to [[papers/architectures/rnn-encoder-decoder|RNN Encoder-Decoder]], [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], [[papers/architectures/mamba|Mamba]], and [[concepts/architectures/state-space-model|State-space model]].

## Limitations

- Sequential recurrence limits training parallelism over time.
- The hidden/cell state is a fixed-size bottleneck for long sequences.
- Long-range retention still has to be learned by gates.
- Large-scale language modeling moved toward attention-based architectures because token states remain directly accessible.
- Comparisons against modern sequence models are sensitive to hardware, implementation, and training budget.
- For graph, image, and molecular structure inputs, LSTM alone does not encode permutation, locality, or equivariance without additional modeling choices.

## Practical Reading Checklist

When using LSTM as a reference point, ask:

- What information must persist across time?
- Does the task require direct access to many previous tokens, or is a compressed state enough?
- Is streaming/bounded-memory inference more important than full-context attention?
- Are gates being used for memory, routing, or simple nonlinear feature selection?
- Does the paper compare against a properly tuned recurrent baseline?
- Does the experiment isolate architecture from objective and data changes?

## Connections

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/attention|Attention]]
- [[papers/architectures/rnn-encoder-decoder|RNN Encoder-Decoder]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/index|Architecture papers]]
