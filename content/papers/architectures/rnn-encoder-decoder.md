---
title: Learning Phrase Representations using RNN Encoder-Decoder
aliases:
  - papers/gru
  - papers/rnn-encoder-decoder
tags:
  - papers
  - architectures
  - recurrent
---

# Learning Phrase Representations using RNN Encoder-Decoder

> The paper introduced a gated recurrent encoder-decoder model and popularized the GRU-style recurrent unit.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation |
| Authors | Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio |
| Year | 2014 |
| Venue | EMNLP 2014 |
| arXiv | [1406.1078](https://arxiv.org/abs/1406.1078) |
| ACL Anthology | [D14-1179](https://aclanthology.org/D14-1179/) |
| Status | verified |

## Question

Phrase-based statistical machine translation used hand-engineered phrase tables and scores. The paper asks whether a neural encoder-decoder can learn continuous phrase representations and phrase-pair probabilities that improve translation scoring.

The architecture question is broader than the SMT system:

$$
\text{Can a variable-length source sequence be compressed into a vector and decoded into a variable-length target sequence?}
$$

This is one of the important pre-attention sequence-to-sequence papers. It sits between classic [[concepts/architectures/rnn|RNN]] language models and later attention-based encoder-decoder systems.

## Main Claim

An RNN encoder-decoder with gated hidden-state updates can learn phrase representations and conditional phrase probabilities useful for translation.

The model factorizes target phrase probability autoregressively:

$$
p(y_1,\ldots,y_{T_y}\mid x_1,\ldots,x_{T_x})
=
\prod_{t=1}^{T_y}
p(y_t\mid y_{<t}, c)
$$

where $c$ is the fixed-length context vector produced by the encoder.

The durable architecture claim is:

$$
\operatorname{Encoder}(x_{1:T_x}) \to c,
\qquad
\operatorname{Decoder}(y_{<t}, c) \to p(y_t\mid y_{<t},c).
$$

The paper also introduces the gated recurrent update now commonly associated with [[concepts/architectures/gru|GRU]]-style recurrence.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | source phrase token sequence $x_{1:T_x}$ |
| Output | target phrase token sequence probability |
| Encoder | recurrent network that compresses source phrase into context vector $c$ |
| Decoder | recurrent network conditioned on $c$ and previous target tokens |
| State | hidden vector, no separate LSTM-style cell state |
| Token mixing | causal recurrence over source and target positions |
| Bridge | fixed-length vector bottleneck |
| Main task | phrase-pair scoring inside an SMT pipeline |
| Later role | reference architecture for seq2seq before attention |

The encoder reads the source phrase:

$$
h_t = f_{\text{enc}}(x_t, h_{t-1})
$$

and produces:

$$
c = h_{T_x}.
$$

The decoder predicts target tokens:

$$
s_t = f_{\text{dec}}(y_{t-1}, s_{t-1}, c)
$$

$$
p(y_t\mid y_{<t},c)=\operatorname{softmax}(g(s_t,c,y_{t-1})).
$$

This is the fixed-vector encoder-decoder pattern. The entire source phrase must pass through $c$.

## GRU-Style Recurrent Unit

The recurrent unit uses gates to control hidden-state updates:

$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1})
$$

$$
\tilde{h}_t
=
\tanh(W_h x_t + U_h(r_t \odot h_{t-1}))
$$

$$
h_t
=
(1-z_t)\odot h_{t-1}
+
z_t\odot \tilde{h}_t.
$$

where:

| Symbol | Meaning |
| --- | --- |
| $z_t$ | update gate |
| $r_t$ | reset gate |
| $\tilde{h}_t$ | candidate hidden state |
| $h_t$ | new hidden state |
| $\odot$ | elementwise product |

The update gate interpolates old memory and new candidate state:

$$
h_t = h_{t-1} + z_t\odot(\tilde{h}_t-h_{t-1}).
$$

This makes the GRU look like a learned residual update over time.

## Gate Interpretation

| Gate | Equation | Role | Failure Mode |
| --- | --- | --- | --- |
| Update gate | $z_t=\sigma(\cdot)$ | controls how much new candidate replaces old state | too low freezes state; too high overwrites memory |
| Reset gate | $r_t=\sigma(\cdot)$ | controls how much previous state enters candidate computation | too low forgets useful context; too high may carry irrelevant state |
| Candidate | $\tilde{h}_t=\tanh(\cdot)$ | proposes replacement content | bottlenecked by hidden size and recurrence |

Compared with [[papers/architectures/long-short-term-memory|LSTM]], the GRU merges hidden and memory state:

| Aspect | GRU | LSTM |
| --- | --- | --- |
| State | one hidden state $h_t$ | hidden state $h_t$ and cell state $c_t$ |
| Gates | update, reset | input, forget, output |
| Parameters | usually fewer | usually more |
| Memory path | interpolated hidden update | protected additive cell path |
| Baseline role | compact gated recurrence | canonical gated memory recurrence |

The paper matters because it made gated recurrence simple enough to become a standard seq2seq component.

## Encoder-Decoder Decomposition

The paper uses the [[concepts/architectures/encoder-decoder|encoder-decoder]] pattern:

$$
c = E_\theta(x_{1:T_x})
$$

$$
p_\phi(y_{1:T_y}\mid x_{1:T_x})
=
\prod_{t=1}^{T_y}
p_\phi(y_t\mid y_{<t}, c).
$$

This decomposition separates representation building from conditional generation.

| Component | Input | Output | Architecture Risk |
| --- | --- | --- | --- |
| Encoder RNN | source phrase tokens | context vector $c$ | fixed-vector bottleneck |
| Decoder RNN | previous target tokens and $c$ | next-token distribution | error accumulation |
| Phrase score | source-target phrase pair | conditional probability | tied to SMT pipeline |
| Gated unit | token and hidden state | updated hidden state | sequential bottleneck |

The later Transformer encoder-decoder keeps the broad decomposition but changes the bridge: instead of a single fixed vector, the decoder uses cross-attention over all encoder token states.

## Fixed-Vector Bottleneck

The encoder compresses a sequence into one vector:

$$
c=h_{T_x}.
$$

This is simple, but it forces all source information through a fixed-size bottleneck. For short phrases this can work well. For long sentences, it becomes a limitation:

$$
I(x_{1:T_x}; c)
\quad\text{is limited by the dimension and training dynamics of}\quad c.
$$

Attention-based models addressed this by exposing all source states:

$$
H = (h_1,\ldots,h_{T_x})
$$

and letting the decoder query them at each target step:

$$
\operatorname{context}_t
=
\operatorname{Attention}(s_t,H,H).
$$

This paper is therefore a key reference for understanding why attention mattered.

## Conditional Probability And Training Objective

The decoder predicts one target token at a time:

$$
p(y_t\mid y_{<t},c)
=
\operatorname{softmax}(o_t).
$$

The negative log-likelihood objective is:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T_y}
\log p(y_t\mid y_{<t},c).
$$

In the original phrase-scoring setup, the learned conditional probability becomes a feature in a larger SMT system. That means the reported translation improvement is not only an architecture claim; it is also a system-integration claim.

## Evidence Reading

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Encoder-decoder scores help SMT | phrase-based SMT experiments | learned phrase scores can improve a translation system | tied to pre-neural SMT pipelines |
| Gated recurrence learns phrase representations | qualitative embedding analysis | phrase embeddings capture syntactic/semantic similarity | qualitative evidence is not enough alone |
| Variable-length phrases can be encoded and decoded | phrase modeling setup | recurrent encoder-decoder handles variable lengths | fixed vector is weak for long sequences |
| GRU-style gating is useful | architecture and empirical performance | simple gated recurrence is practical | not a direct universal comparison to LSTM |

Read the paper as two contributions:

1. gated recurrent unit architecture;
2. encoder-decoder phrase scoring for SMT.

These should not be collapsed into one claim.

## Relation To LSTM

[[papers/architectures/long-short-term-memory|LSTM]] introduced a protected cell state:

$$
c_t = f_t\odot c_{t-1}+i_t\odot\tilde{c}_t.
$$

GRU uses a single hidden state:

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t.
$$

Both use gates and additive/interpolated update paths. The GRU is simpler; LSTM has a more explicit memory cell. When reading architecture papers, compare them by:

- hidden dimension and parameter count;
- training budget and optimizer;
- sequence length;
- reset/carry policy across sequences;
- whether bidirectionality is used;
- how final representations are pooled.

## Relation To Attention

The fixed-vector encoder-decoder computes:

$$
c=E(x_{1:T_x})
$$

and every decoder step sees the same $c$.

Attention changes this to:

$$
c_t = \sum_{i=1}^{T_x}\alpha_{ti}h_i
$$

where:

$$
\alpha_{ti}
=
\operatorname{softmax}_i(\operatorname{score}(s_t,h_i)).
$$

This makes the source representation step-dependent. The decoder can align each target token with relevant source positions instead of relying on one compressed vector.

This transition leads directly toward [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], where recurrence is removed and attention becomes the main token-mixing mechanism.

## Relation To Transformer Encoder-Decoder

The high-level contract remains:

$$
\text{source} \to \text{encoder} \to \text{decoder} \to \text{target}.
$$

But the implementation changes:

| Dimension | RNN Encoder-Decoder | Transformer Encoder-Decoder |
| --- | --- | --- |
| Encoder token mixing | recurrence | self-attention |
| Decoder token mixing | recurrence | masked self-attention |
| Source-target bridge | fixed vector $c$ | cross-attention over encoder states |
| Parallelism in training | sequential over time | token-parallel under teacher forcing |
| Long-range path | hidden-state bottleneck | direct attention path |
| Position handling | recurrence order | positional encoding |

This is why the paper belongs in an architecture shelf even though the application is phrase-based SMT.

## Implementation Notes

Important details for reproducing or comparing this family:

| Detail | Why It Matters |
| --- | --- |
| tokenization | phrase vocabulary and rare words dominate behavior |
| hidden size | controls fixed-vector bottleneck capacity |
| teacher forcing | training uses ground-truth previous target tokens |
| decoding strategy | greedy, beam, or phrase-table integration changes system output |
| state initialization | decoder initial state from encoder context changes conditioning |
| padding masks | padding must not update recurrent state incorrectly |
| bidirectionality | changes encoder context and fairness of comparison |
| phrase scoring integration | improvement may come from system feature weighting |

For a standalone seq2seq model, the output probability is enough. For SMT integration, the neural score becomes one feature among others, so the full system objective matters.

## Common Misreadings

### "This paper is only about GRU"

It is also an encoder-decoder phrase representation paper. The GRU-style unit became the most reusable architecture component, but the system contribution is phrase scoring for SMT.

### "Encoder-decoder means Transformer"

No. Encoder-decoder is a broad architectural decomposition. Transformers are one implementation; recurrent encoder-decoders came earlier.

### "The fixed vector is just an implementation detail"

The fixed vector is the central bridge and the central limitation. It defines what information the decoder can access.

### "Better translation means the recurrent unit alone is better"

The reported translation result depends on data, phrase-table integration, objective, and decoding pipeline.

## What To Check In Later Seq2Seq Papers

- Does the encoder expose a single vector or all token states?
- Is attention used?
- Is the decoder autoregressive?
- Does training use teacher forcing?
- How is inference decoded: greedy, beam, sampling, constrained search?
- Are results from a standalone neural model or a hybrid SMT/NMT system?
- Are GRU/LSTM/Transformer baselines matched in parameters and training budget?
- Is sequence length long enough to test the bottleneck?

## Why It Still Matters

This paper is a canonical reference for:

- GRU-style gated recurrence;
- fixed-vector encoder-decoder sequence modeling;
- phrase representation learning;
- the pre-attention limitation that motivated attention;
- the transition from statistical machine translation to neural sequence models.

For this wiki, it connects [[papers/architectures/long-short-term-memory|LSTM]], [[concepts/architectures/gru|GRU]], [[concepts/architectures/encoder-decoder|Encoder-decoder]], [[concepts/architectures/attention|Attention]], and [[papers/architectures/attention-is-all-you-need|Transformer]].

## Limitations

- Fixed-length context vector creates a bottleneck for long inputs.
- Sequential recurrence limits training parallelism.
- Phrase-level experiments do not directly cover full modern sequence generation.
- SMT system integration makes architecture attribution less clean.
- GRU and LSTM comparisons are sensitive to parameter matching and tuning.
- Attention-based models quickly became stronger for translation.

## Connections

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
