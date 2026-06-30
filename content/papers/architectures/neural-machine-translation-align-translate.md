---
title: Neural Machine Translation by Jointly Learning to Align and Translate
aliases:
  - papers/bahdanau-attention
  - papers/neural-machine-translation-by-jointly-learning-to-align-and-translate
  - papers/additive-attention
tags:
  - papers
  - architectures
  - attention
  - sequence-modeling
  - machine-translation
---

# Neural Machine Translation by Jointly Learning to Align and Translate

> The paper introduced neural attention as a soft alignment mechanism that lets a decoder look back at source-token states instead of relying on one fixed-length sentence vector.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Machine Translation by Jointly Learning to Align and Translate |
| Authors | Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio |
| Year | 2014 preprint; 2015 conference |
| Venue | ICLR 2015 |
| arXiv | [1409.0473](https://arxiv.org/abs/1409.0473) |
| ICLR PDF | [conference archive](https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia%3Diclr2015%3Abahdanau-iclr2015.pdf) |
| Status | full note started |

## One-Line Takeaway

Bahdanau attention replaces the fixed encoder context vector with a learned soft alignment over encoder states for each decoder step.

## Question

Early neural machine translation used an encoder-decoder architecture:

$$
x_{1:T_x}
\rightarrow
c
\rightarrow
y_{1:T_y}.
$$

The encoder compresses the full source sentence into one fixed-length vector:

$$
c = f_{\text{enc}}(x_{1:T_x}).
$$

The decoder generates:

$$
p(y_t\mid y_{<t},c).
$$

The bottleneck:

$$
\text{long source sentence}
\rightarrow
\text{single vector }c.
$$

The paper asks:

> Can the decoder learn where to look in the source sentence while generating each target word?

## Main Claim

The narrowed architecture claim:

$$
\text{encoder states}
+
\text{decoder state}
+
\text{learned soft alignment}
\Rightarrow
\text{better sequence-to-sequence translation}.
$$

Instead of one context vector for the entire source sentence:

$$
c,
$$

the model uses a different context vector at every decoder step:

$$
c_i
=
\sum_{j=1}^{T_x}
\alpha_{ij}h_j.
$$

Here:

- $i$ indexes target position;
- $j$ indexes source position;
- $h_j$ is an encoder annotation/state;
- $\alpha_{ij}$ is the learned attention weight.

## Architecture Contract

| Component | Role |
| --- | --- |
| bidirectional encoder | builds source annotations $h_j$ |
| decoder RNN | generates target tokens autoregressively |
| alignment model | scores decoder state against source states |
| softmax over source positions | normalizes scores into attention weights |
| context vector $c_i$ | weighted sum of source annotations for target step $i$ |
| output model | predicts next target token from decoder state and context |

This is the architecture pattern:

$$
\text{source sequence}
\rightarrow
\{h_1,\ldots,h_{T_x}\}
$$

$$
(s_{i-1}, \{h_j\})
\rightarrow
\alpha_{ij}
\rightarrow
c_i
\rightarrow
y_i.
$$

## Encoder Annotations

The encoder produces a sequence of annotations:

$$
H = (h_1,\ldots,h_{T_x}).
$$

The paper uses a bidirectional RNN. A simplified form:

$$
\overrightarrow{h}_j
=
\overrightarrow{f}(x_j,\overrightarrow{h}_{j-1}),
$$

$$
\overleftarrow{h}_j
=
\overleftarrow{f}(x_j,\overleftarrow{h}_{j+1}).
$$

The annotation concatenates both directions:

$$
h_j
=
\left[
\overrightarrow{h}_j;
\overleftarrow{h}_j
\right].
$$

Why bidirectional encoding matters:

$$
h_j
\text{ can represent source word }j
\text{ with left and right context}.
$$

The decoder does not attend to raw tokens. It attends to contextual source annotations.

## Decoder State

The decoder is an autoregressive RNN:

$$
s_i
=
f(s_{i-1}, y_{i-1}, c_i).
$$

The token distribution:

$$
p(y_i\mid y_{<i},x)
=
g(y_{i-1},s_i,c_i).
$$

This means generation still proceeds left to right:

$$
p(y_{1:T_y}\mid x)
=
\prod_{i=1}^{T_y}
p(y_i\mid y_{<i},x).
$$

Attention changes the context interface, not the autoregressive factorization.

## Alignment Model

The alignment score compares decoder state and source annotation:

$$
e_{ij}
=
a(s_{i-1},h_j).
$$

A common additive attention form:

$$
e_{ij}
=
v_a^\top
\tanh
\left(
W_s s_{i-1}
+
W_h h_j
\right).
$$

Then:

$$
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_{k=1}^{T_x}
\exp(e_{ik})
}.
$$

The attention weights form a distribution over source positions:

$$
\sum_{j=1}^{T_x}
\alpha_{ij}
=
1.
$$

Reading:

$$
\alpha_{ij}
\approx
\text{how much target position }i
\text{ uses source position }j.
$$

But this should be read as a model-internal soft alignment, not guaranteed causal explanation.

## Context Vector

For target position $i$, the context vector is:

$$
c_i
=
\sum_{j=1}^{T_x}
\alpha_{ij}h_j.
$$

This is the core architecture move. The context is no longer fixed:

$$
c
\rightarrow
c_i.
$$

Each output step can use a different weighted mixture of source states.

The model therefore changes from:

$$
p(y_i\mid y_{<i},c)
$$

to:

$$
p(y_i\mid y_{<i},c_i),
\qquad
c_i=c_i(s_{i-1},H).
$$

## Fixed-Vector Bottleneck

The earlier encoder-decoder route:

$$
x_{1:T_x}
\rightarrow
c
\rightarrow
y_{1:T_y}
$$

forces all source information through one vector.

This is especially harmful for long sentences:

$$
T_x \uparrow
\Rightarrow
\text{compression burden increases}.
$$

Bahdanau attention changes the memory interface:

$$
x_{1:T_x}
\rightarrow
(h_1,\ldots,h_{T_x})
\rightarrow
\text{decoder reads from source states}.
$$

The decoder now has differentiable random access to source positions.

## Soft Alignment

Traditional machine translation used alignment ideas explicitly. Bahdanau attention learns soft alignments inside the neural model.

Hard alignment would choose one source position:

$$
j^\star_i
=
\operatorname*{arg\,max}_j e_{ij}.
$$

Soft alignment uses a distribution:

$$
\alpha_i
=
(\alpha_{i1},\ldots,\alpha_{iT_x}).
$$

Then the context vector is differentiable:

$$
c_i
=
\sum_j
\alpha_{ij}h_j.
$$

This is why the alignment can be trained end to end by backpropagation from the translation loss.

## Training Objective

The model maximizes conditional log likelihood:

$$
\mathcal{L}
=
\sum_{(x,y)}
\log p_\theta(y\mid x).
$$

With autoregressive factorization:

$$
\log p_\theta(y\mid x)
=
\sum_{i=1}^{T_y}
\log p_\theta(y_i\mid y_{<i},x).
$$

The attention parameters are trained through the same objective:

$$
\nabla_\theta
\log p_\theta(y_i\mid y_{<i},c_i).
$$

There is no separate supervised alignment label required.

## Why It Matters for Attention

The paper made attention a reusable architectural primitive:

$$
\text{query}
+
\text{memory}
\rightarrow
\text{weighted read}.
$$

In Bahdanau attention:

| Role | Object |
| --- | --- |
| query | decoder state $s_{i-1}$ |
| keys | source annotations $h_j$ through scoring network |
| values | source annotations $h_j$ |
| weights | $\alpha_{ij}$ |
| readout | context vector $c_i$ |

This is not yet Transformer self-attention, but it establishes the central idea:

$$
\text{compute relevance}
\rightarrow
\text{normalize}
\rightarrow
\text{weighted sum}.
$$

## Additive vs Dot-Product Attention

Bahdanau attention is often called additive attention because the score uses a learned MLP:

$$
e_{ij}
=
v_a^\top
\tanh(W_s s_{i-1}+W_h h_j).
$$

Dot-product attention uses:

$$
e_{ij}
=
q_i^\top k_j.
$$

Scaled dot-product attention uses:

$$
e_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}.
$$

Reading:

| Attention Type | Score |
| --- | --- |
| additive | learned MLP over query and key |
| dot-product | inner product similarity |
| scaled dot-product | normalized inner product for stability |

The Transformer later uses scaled dot-product attention, but the alignment-as-soft-read idea is already present here.

## Relation to RNN Encoder-Decoder

[[papers/architectures/rnn-encoder-decoder|RNN Encoder-Decoder]]:

$$
x_{1:T_x}
\rightarrow
c
\rightarrow
y_{1:T_y}.
$$

Bahdanau attention:

$$
x_{1:T_x}
\rightarrow
(h_1,\ldots,h_{T_x})
\rightarrow
(c_1,\ldots,c_{T_y})
\rightarrow
y_{1:T_y}.
$$

The difference is not only performance. It changes the interface from fixed vector compression to memory access.

## Relation to Transformer

Transformer attention:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

Bahdanau attention:

$$
c_i
=
\sum_j
\operatorname{softmax}_j(a(s_{i-1},h_j))h_j.
$$

The shared abstraction:

$$
\text{query-specific weighted sum over values}.
$$

The differences:

| Axis | Bahdanau Attention | Transformer Attention |
| --- | --- | --- |
| backbone | RNN encoder-decoder | attention-only blocks |
| score | additive MLP | scaled dot product |
| query | decoder state | projected token states |
| memory | encoder annotations | keys and values |
| parallelism | decoder remains recurrent | self-attention highly parallelizable |
| role | remove fixed-context bottleneck | replace recurrence as sequence mixer |

## Relation to Cross-Attention

Bahdanau attention is an early cross-attention pattern:

$$
\text{decoder query}
\rightarrow
\text{source memory}.
$$

Modern cross-attention:

$$
\operatorname{CrossAttn}(X,C)
=
\operatorname{softmax}
\left(
\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}
\right)CW_V.
$$

The direction matters:

$$
\text{target-side decoder}
\quad
\text{queries}
\quad
\text{source-side encoder states}.
$$

This idea later appears in encoder-decoder Transformers, retrieval-augmented models, multimodal generation, and protein-ligand context mixing.

## Evidence Reading

The paper evaluates English-to-French translation and analyzes learned alignments.

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| translation improvement | attention helps encoder-decoder NMT | attention is universally sufficient |
| stronger long-sentence behavior | fixed-vector bottleneck was real | all long-context problems are solved |
| alignment visualization | learned weights often match intuitive alignments | attention weights are causal explanations |
| end-to-end training | alignments can emerge without labels | learned alignments are always faithful |

The key evidence is architectural: dynamic context improves sequence transduction, especially when fixed-length encoding is a bottleneck.

## Attention Weights and Explanation

Attention weights:

$$
\alpha_{ij}
$$

show how the context vector is computed:

$$
c_i=\sum_j\alpha_{ij}h_j.
$$

But they do not automatically prove:

$$
\alpha_{ij}
\Rightarrow
\text{source token }j\text{ caused target token }i.
$$

Why:

- source annotations already mix context;
- decoder state contains previous target history;
- downstream nonlinearities transform the context;
- alternative attention patterns can sometimes yield similar outputs.

The correct statement is:

$$
\alpha_{ij}
\text{ is a model-internal soft read weight}.
$$

## Implementation Notes

Implementation checklist:

| Component | Check |
| --- | --- |
| encoder annotations | are source states returned for all positions? |
| decoder query | does score use previous decoder state or current state? |
| score shape | $e\in\mathbb{R}^{T_y\times T_x}$ |
| softmax axis | normalize over source positions |
| context vector | weighted sum over source annotations |
| masking | ignore source padding positions |
| training | teacher forcing and target autoregressive likelihood |

Shape sketch:

$$
H\in\mathbb{R}^{T_x\times d_h},
\qquad
s_i\in\mathbb{R}^{d_s}.
$$

Scores:

$$
e_i\in\mathbb{R}^{T_x}.
$$

Weights:

$$
\alpha_i=\operatorname{softmax}(e_i).
$$

Context:

$$
c_i=H^\top\alpha_i
\in\mathbb{R}^{d_h}.
$$

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| no source padding mask | attention can read padded tokens |
| softmax over wrong axis | weights normalize over target instead of source |
| fixed context still used incorrectly | model may not actually use dynamic memory |
| overinterpreting alignments | attention weights are not causal proof |
| long-source quadratic-ish score cost | each target step scores all source states |
| decoder recurrence bottleneck | attention helps memory but does not remove sequential decoding |
| exposure bias | teacher-forced training differs from autoregressive inference |

## Common Misreadings

### "This is the Transformer attention paper."

No. This is the neural attention/alignment paper for RNN encoder-decoder translation. Transformer later replaces recurrent sequence modeling with stacked self-attention and feed-forward blocks.

### "Attention was invented only for interpretability."

No. The primary architectural role is dynamic memory access. Interpretability is a secondary and limited reading.

### "The context vector disappears."

No. The fixed global context is replaced by a target-step-specific context vector $c_i$.

### "Attention weights are hard alignments."

No. They are soft differentiable distributions over source positions.

## Later-Paper Checklist

When reading later attention papers, ask:

- What is the query object?
- What provides keys and values?
- Is the attention additive, dot-product, scaled dot-product, or structured?
- Is attention self-attention or cross-attention?
- What axis is normalized?
- What masks are applied?
- Does attention replace recurrence or only augment it?
- Are weights interpreted as explanation, alignment, or only mixing coefficients?
- Does the model use one context vector or dynamic context per step?
- How does complexity scale with source and target lengths?

## Why It Matters

This paper is the bridge:

$$
\text{RNN encoder-decoder}
\rightarrow
\text{attention-based sequence modeling}
\rightarrow
\text{Transformer}.
$$

It made a reusable pattern explicit:

$$
\text{learn where to read}
\quad
\text{while generating}.
$$

For this wiki, it should sit between RNN Encoder-Decoder and Attention Is All You Need because it explains why attention became necessary before it became the whole architecture.

## Limitations

The paper still uses recurrent encoder-decoder machinery. Attention removes the fixed-vector bottleneck, but not all limitations:

- target decoding remains sequential;
- source-target scoring grows with source and target lengths;
- learned alignments are soft and not guaranteed faithful;
- translation quality still depends on data, vocabulary, optimization, and decoding;
- attention is a memory access mechanism, not a full architecture by itself.

The defensible claim:

$$
\text{Bahdanau attention}
\Rightarrow
\text{dynamic differentiable source memory access for sequence transduction}.
$$

The overclaim to avoid:

$$
\text{attention weights}
\Rightarrow
\text{complete explanation of model behavior}.
$$

## Connections

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/tasks/sequence-to-sequence|Sequence-to-sequence]]
- [[papers/architectures/rnn-encoder-decoder|RNN Encoder-Decoder]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/t5|T5]]
- [[papers/architectures/index|Architecture papers]]
