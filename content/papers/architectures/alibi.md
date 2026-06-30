---
title: ALiBi
aliases:
  - papers/alibi
  - papers/attention-with-linear-biases
  - papers/train-short-test-long
tags:
  - papers
  - architectures
  - transformer
  - attention
  - positional-encoding
  - long-context
---

# ALiBi

> ALiBi injects position into attention by adding a head-specific linear distance penalty to attention logits instead of adding positional embeddings to token representations.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation |
| Authors | Ofir Press, Noah A. Smith, Mike Lewis |
| Year | 2021 preprint; 2022 ICLR paper |
| Venue | ICLR 2022 |
| arXiv | [2108.12409](https://arxiv.org/abs/2108.12409) |
| Status | full note started |

## One-Line Takeaway

ALiBi turns positional information into an additive bias on attention scores:

$$
A_{h,i,j}
=
\frac{q_{h,i}^{\top}k_{h,j}}{\sqrt{d_k}}
+ m_h b_{i,j},
$$

where $m_h$ is a fixed slope for head $h$ and $b_{i,j}$ is a distance-based bias. In causal language modeling, distant past tokens get a larger penalty.

## Question

Plain self-attention has no order signal unless position is injected somewhere:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

The usual [[papers/architectures/attention-is-all-you-need|Transformer]] route adds positional encodings to token embeddings:

$$
x_i
=
e_i+p_i.
$$

That works, but long-context extrapolation raises a sharper question:

$$
\text{Can a model trained on length }L_{\text{train}}
\text{ run well at }L_{\text{test}}>L_{\text{train}}?
$$

ALiBi asks whether position should enter the attention logit directly rather than the token embedding.

## Main Claim

The paper's architecture claim is:

$$
\text{linear relative-position bias in attention logits}
\rightarrow
\text{length extrapolation without learned position embeddings}.
$$

Instead of learning or tabulating one position vector per index, ALiBi uses a fixed monotonic bias:

$$
\text{near tokens}
>
\text{far tokens}
$$

in the attention score before softmax.

For a causal query at position $i$ attending to a key at position $j\leq i$, define distance:

$$
d(i,j)=i-j.
$$

Then a simplified ALiBi logit is:

$$
A_{h,i,j}
=
\frac{q_{h,i}^{\top}k_{h,j}}{\sqrt{d_k}}
- m_h d(i,j).
$$

The slope $m_h>0$ differs across heads, giving some heads a strong recency bias and others a weaker long-range bias.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | ordered token sequence |
| Target block | self-attention logits |
| Position operation | add fixed head-specific linear bias by relative distance |
| Token embedding | no added positional embedding required by ALiBi |
| Parameters | no learned position table |
| Main use | decoder-only language modeling and long-context extrapolation |
| Main claim | train on shorter sequences and test on longer sequences |
| Main risk | bias shape is fixed and may not fit every task or attention pattern |

ALiBi is not a new backbone. It is a positional mechanism for attention.

## Attention Logit View

For a batch and one head, standard attention logits are:

$$
S_{i,j}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}.
$$

With a causal mask $M_{i,j}$:

$$
A_{i,j}
=
S_{i,j}+M_{i,j}.
$$

ALiBi adds a relative-position term:

$$
A_{h,i,j}
=
S_{h,i,j}
+M_{i,j}
+B_{h,i,j}.
$$

For causal attention:

$$
B_{h,i,j}
=
-m_h(i-j)
\qquad
\text{for }j\leq i.
$$

Then:

$$
P_{h,i,:}
=
\operatorname{softmax}(A_{h,i,:}).
$$

The value mixing is unchanged:

$$
y_{h,i}
=
\sum_{j\leq i}P_{h,i,j}v_{h,j}.
$$

ALiBi only changes the logit geometry before softmax.

## Why Linear Bias Helps Extrapolation

Learned absolute embeddings usually define parameters only for positions seen during training:

$$
p_1,\ldots,p_{L_{\text{train}}}.
$$

Extrapolating to a longer sequence requires some strategy for:

$$
p_{L_{\text{train}}+1},\ldots,p_{L_{\text{test}}}.
$$

ALiBi avoids this lookup-table issue. The distance penalty is defined for any distance:

$$
d\in\{0,1,2,\ldots\}.
$$

So the same rule extends beyond the training length:

$$
B(d)=-m_hd.
$$

This does not guarantee perfect long-context behavior, but it removes one hard length boundary from the architecture.

## Head Slopes

ALiBi uses different slopes for different heads:

$$
m_1,m_2,\ldots,m_H.
$$

The architecture intuition:

| Head Bias | Effect |
| --- | --- |
| large slope | strong preference for nearby tokens |
| small slope | weaker distance penalty and more long-range access |

This creates a spread of attention distance priors across heads without learning a position table.

## Relation to Other Position Methods

| Method | Where Position Enters | Extrapolation Story |
| --- | --- | --- |
| sinusoidal position | add fixed vector to token embedding | analytic but still mixed into token state |
| learned absolute position | add learned vector to token embedding | tied to trained position range |
| relative position bias | add learned or bucketed bias to logits | depends on buckets and maximum distance |
| [RoPE](/papers/architectures/roformer) | rotate query/key vectors | relative phase, later often scaled for longer context |
| ALiBi | add fixed linear distance bias to logits | distance rule naturally extends |

The key distinction:

$$
\text{RoPE changes }Q,K;
\qquad
\text{ALiBi changes logits}.
$$

Both are position mechanisms for [[concepts/architectures/attention|Attention]], but their implementation and extrapolation behavior differ.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| train-short test-long works | language modeling perplexity at longer test lengths | ALiBi can extrapolate beyond training context | result depends on task and model setup |
| memory and speed improve | comparison against training at longer context length | shorter training sequences can reduce training cost | not a free serving-time long-context method |
| recency bias helps | comparison with position methods on WikiText-103 | fixed distance prior is useful for language modeling | recency prior may not fit every modality |
| implementation is small | logit-bias addition | easy to add to existing attention code | kernel and mask code must handle bias correctly |

## Implementation Reading

An implementation should make these decisions explicit:

- Is the model causal, bidirectional, or cross-attentional?
- Are distances $i-j$, $|i-j|$, or another relation?
- Are slopes fixed exactly as in the paper or changed?
- Is the ALiBi bias added before softmax and before masking is finalized?
- Does the attention kernel support additive bias without materializing a huge dense matrix?
- Are train and test context lengths reported separately?

For causal language models, the simplified logit update is:

$$
\texttt{logits}
\leftarrow
\texttt{logits}
+ \texttt{causal\_mask}
+ \texttt{alibi\_bias}.
$$

## Limitations

- ALiBi is a fixed inductive bias; it is not universally optimal for every sequence task.
- It helps remove positional-table length limits, but attention compute still scales with context length unless paired with efficient attention or sparsity.
- Extrapolation claims should report both training length and evaluation length.
- Some later long-context systems prefer RoPE variants or position interpolation/scaling, so ALiBi is one branch of the design space rather than the final answer.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "ALiBi makes attention linear time." | It changes position bias, not dense attention complexity. |
| "ALiBi is a new Transformer architecture." | It is a positional mechanism inside attention. |
| "No positional embeddings means no position information." | Position enters through the logit bias. |
| "Length extrapolation is solved completely." | The paper shows a useful route; evaluation still depends on task, data, model size, and context range. |

## What to Remember

ALiBi is the cleanest example of position as an attention-logit prior:

$$
\text{attention score}
=
\text{content similarity}
+ \text{distance bias}.
$$

For the architecture shelf, keep it near [[papers/architectures/roformer|RoPE]], [[papers/architectures/transformer-xl|Transformer-XL]], [[papers/architectures/performer|Performer]], and [[papers/architectures/flashattention|FlashAttention]] because it changes how Transformers handle sequence length without changing the whole backbone.

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/roformer|RoFormer]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/flashattention|FlashAttention]]
