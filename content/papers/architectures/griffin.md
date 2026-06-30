---
title: Griffin
aliases:
  - papers/griffin
  - papers/hawk-griffin
  - papers/mixing-gated-linear-recurrences-local-attention
tags:
  - papers
  - architectures
  - recurrent
  - attention
  - efficient-attention
  - sequence-modeling
  - language-models
---

# Griffin

> Griffin is important because it treats recurrent layers and local attention as complementary building blocks for efficient language models, rather than asking whether one family should fully replace the other.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models |
| Authors | Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, Caglar Gulcehre |
| Year | 2024 |
| Venue | arXiv preprint |
| arXiv | [2402.19427](https://arxiv.org/abs/2402.19427) |
| Status | full note started |

## Question

Transformers are easy to train in parallel, but attention creates long-context costs:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

Recurrent models can decode with fixed-size state:

$$
h_t
=
f_\theta(h_{t-1},x_t),
$$

but classic RNNs were harder to scale and train effectively at language-model scale.

Griffin asks:

$$
\text{Can a modern recurrent block be scaled like a Transformer while using local attention only where attention is most useful?}
$$

The paper proposes:

| Model | Temporal Mixer |
| --- | --- |
| Hawk | gated linear recurrence |
| Griffin | gated linear recurrence plus local attention |

## Main Claim

The durable architecture claim is:

$$
\text{gated linear recurrence}
+
\text{local attention}
\Rightarrow
\text{efficient language-model backbone}.
$$

The paper reports that Hawk and Griffin scale with training compute, that Griffin can be scaled to 14B parameters, and that Griffin has lower latency and higher throughput during inference than Transformer baselines in reported settings.

For this wiki, the key lesson is:

$$
\text{hybrid architectures can allocate attention only where full token-token interaction is most valuable}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Output | next-token distribution |
| Baseline | MQA Transformer language model |
| Recurrent block | gated linear recurrence, called RG-LRU in the paper |
| Hybrid block | local attention mixed with recurrent blocks |
| Attention scope | local/sliding-window MQA rather than full global attention everywhere |
| Main goal | Transformer-like trainability with recurrent inference efficiency |
| Scaling target | language models up to 14B parameters |

The objective is still causal language modeling:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
$$

The architecture change is in the temporal mixing block.

## Residual Block View

Griffin keeps a Transformer-like residual scaffold:

$$
\tilde{x}
=
x
+
\operatorname{TemporalMix}(\operatorname{RMSNorm}(x)),
$$

$$
y
=
\tilde{x}
+
\operatorname{MLP}(\operatorname{RMSNorm}(\tilde{x})).
$$

This is important. Griffin is not a return to an old unnormalized RNN stack. It keeps modern residual, normalization, and MLP design while changing the sequence mixer.

The temporal mixer can be:

| Mixer | Role |
| --- | --- |
| global MQA | Transformer baseline |
| local MQA | local token-token interaction with bounded window |
| RG-LRU | gated recurrent state update |

## Gated Linear Recurrence

A simplified gated linear recurrence is:

$$
h_t
=
a_t\odot h_{t-1}
+
b_t\odot u_t,
$$

where $a_t$ controls retention of previous state, $b_t$ controls writing new information, and $u_t$ is a projected token signal.

The important property is fixed-size recurrent state:

$$
\operatorname{state\ memory}
=
O(d)
\quad
\text{per layer},
$$

rather than storing every previous key and value:

$$
\operatorname{KV\ cache}
=
O(Td)
\quad
\text{per attention layer}.
$$

This is the inference motivation behind Hawk and Griffin.

## Local Attention Path

Local attention restricts token-token interaction to a window:

$$
\mathcal{N}(t)
=
\{i : t-w < i \le t\}.
$$

Then:

$$
y_t
=
\sum_{i\in\mathcal{N}(t)}
\operatorname{softmax}_i(q_t^\top k_i)v_i.
$$

This preserves explicit token retrieval over nearby context without giving every layer a full-context KV cache.

The Griffin design point is:

$$
\text{recurrent state for long streaming context}
\quad+\quad
\text{local attention for precise nearby interactions}.
$$

## Hawk vs Griffin

Hawk and Griffin should be read together.

| Model | Architecture Idea | Main Use in Reading |
| --- | --- | --- |
| Hawk | recurrent-only stack with gated linear recurrence | tests whether modern recurrence alone can scale |
| Griffin | recurrent stack mixed with local attention | tests whether small attention allocation improves the recurrent backbone |

Hawk is the cleaner recurrent hypothesis. Griffin is the practical hybrid hypothesis.

## Relation to Mamba, GLA, and DeltaNet

Griffin belongs in the same family pressure as [[papers/architectures/mamba|Mamba]], [[papers/architectures/gated-linear-attention|GLA]], and [[papers/architectures/deltanet|DeltaNet]]:

$$
\text{reduce attention cost}
\quad+\quad
\text{keep useful sequence memory}
\quad+\quad
\text{retain trainability}.
$$

But Griffin's answer is different:

| Paper | Main Mixer |
| --- | --- |
| [Mamba](/papers/architectures/mamba) | selective state-space scan |
| [GLA](/papers/architectures/gated-linear-attention) | gated linear-attention matrix state |
| [DeltaNet](/papers/architectures/deltanet) | delta-rule matrix memory |
| [Griffin](/papers/architectures/griffin) | gated linear recurrence plus local attention |

This makes Griffin especially useful when reading hybrid backbones such as [[papers/architectures/jamba|Jamba]].

## Relation to Jamba

[[papers/architectures/jamba|Jamba]] mixes Transformer attention, Mamba layers, and MoE.

Griffin mixes local attention and gated linear recurrence.

Both papers support the same broader architecture direction:

$$
\text{do not choose one mixer globally}
\quad\rightarrow\quad
\text{allocate mixers by role}.
$$

The difference is the ingredients:

| Paper | Hybrid Ingredients |
| --- | --- |
| Griffin | local MQA plus gated linear recurrence |
| Jamba | attention plus Mamba plus MoE |

## Evidence to Read Carefully

The paper reports scaling behavior, downstream comparisons, long-sequence extrapolation, inference latency/throughput, and distributed training considerations.

Read each claim separately:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| scaling curves | recurrent/local-attention models can scale with compute | universal superiority over Transformers |
| downstream comparisons | strong reported language-model performance | all tasks or all tokenizer/data choices |
| long-sequence extrapolation | recurrence/local attention can generalize past train length | exact retrieval in all long contexts |
| latency/throughput | inference can benefit from fixed-size state | deployment speed independent of batch/hardware |
| 14B scale | model family can be trained at large scale | open reproducibility or public model availability |

The important limitation is also explicit: pre-trained Hawk/Griffin models can lag Transformers on copying and exact-retrieval tasks without fine-tuning. That matters when comparing recurrent compression to explicit attention.

## Practical Checks

- Count how many layers use recurrence, local attention, or global attention.
- Separate training efficiency from autoregressive inference efficiency.
- Check context length used for training and evaluation.
- Track local attention window size.
- Compare retrieval-heavy tasks separately from average validation loss.
- Check whether claims depend on MQA, RMSNorm, MLP design, data amount, or recurrence alone.
- Treat latency and throughput as hardware, batch-size, and implementation dependent.

## Where It Fits

Griffin fills the hybrid recurrent-attention slot:

$$
\text{Mamba/GLA/DeltaNet}
\rightarrow
\text{recurrent mixer alternatives}
\rightarrow
\text{Griffin}
\rightarrow
\text{hybrid allocation of recurrence and attention}.
$$

It is a useful bridge before reading RecurrentGemma-style model releases or newer hybrid attention-recurrence architectures.

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/gated-linear-attention|Gated Linear Attention]]
- [[papers/architectures/deltanet|DeltaNet]]
- [[papers/architectures/jamba|Jamba]]
