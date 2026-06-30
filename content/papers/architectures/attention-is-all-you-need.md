---
title: Attention Is All You Need
aliases:
  - papers/attention-is-all-you-need
  - papers/transformer-paper
tags:
  - papers
  - architectures
  - transformer
  - attention
---

# Attention Is All You Need

> The paper introduced the Transformer: an encoder-decoder architecture that replaces recurrent and convolutional sequence modeling with attention-based token mixing.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Attention Is All You Need |
| Authors | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1706.03762](https://arxiv.org/abs/1706.03762) |
| NeurIPS | [Proceedings page](https://papers.nips.cc/paper/7181-attention-is-all-you-need) |
| Status | verified |

## Question

Before this paper, strong sequence transduction systems were usually built around recurrent or convolutional encoder-decoder models, often with attention added on top. The question was whether recurrence and convolution were necessary for high-quality sequence modeling, or whether attention alone could do the sequence mixing.

The deeper architecture question is about sequence dependence. RNNs process tokens sequentially, CNNs process local windows with depth-dependent receptive fields, and attention lets every token directly address every other token in one layer. The paper tests whether that direct token-token interaction can become the main computational primitive.

## Main Claim

The paper's central claim is that a sequence transduction model can be built entirely from attention and feed-forward blocks, without recurrence or convolution, while improving translation quality and training parallelism on the reported machine translation benchmarks.

Narrowed claim:

$$
\text{Transformer}
\Rightarrow
\text{strong WMT translation results under the paper's training and evaluation protocol}
$$

This is not the same as proving that recurrence or convolution are never useful. It shows that attention-only sequence modeling is a strong architecture class under the tested settings.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | source token sequence and target prefix sequence |
| Output | next-token distribution over target vocabulary |
| Backbone | stacked encoder blocks and decoder blocks |
| Token mixing | self-attention in encoder and decoder; cross-attention from decoder to encoder |
| Position information | sinusoidal or learned positional encodings |
| Parallelism | encoder tokens and teacher-forced decoder tokens can be processed in parallel during training |
| Causal constraint | decoder self-attention masks future target tokens |

The translation training objective is autoregressive over the target sequence:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T_y}
\log p_\theta(y_t \mid y_{<t}, x)
$$

where $x$ is the source sequence and $y_{<t}$ is the target prefix.

At inference time:

$$
p_\theta(y \mid x)
=
\prod_{t=1}^{T_y}
p_\theta(y_t \mid y_{<t}, x)
$$

The encoder builds source representations; the decoder repeatedly queries those representations while generating target tokens.

## Method

The Transformer uses stacked encoder and decoder blocks. Each block combines:

- [[concepts/architectures/attention|scaled dot-product attention]];
- [[concepts/architectures/transformer|multi-head self-attention]];
- encoder-decoder [[concepts/architectures/cross-attention|cross-attention]] in the decoder;
- position-wise [[concepts/architectures/feed-forward-network|feed-forward networks]];
- [[concepts/architectures/residual-connection|residual connections]];
- [[concepts/architectures/normalization-placement|normalization]];
- [[concepts/architectures/positional-encoding|positional encoding]].

The core attention formula is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

where $Q$ contains queries, $K$ contains keys, $V$ contains values, and $d_k$ is the key dimension. The scale factor reduces extreme dot-product magnitudes before [[concepts/architectures/softmax|softmax]].

Multi-head attention projects the same input into several attention subspaces:

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

Because the architecture has no recurrence, positional information is injected with positional encodings:

$$
\operatorname{PE}_{(pos,2i)}
=
\sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

$$
\operatorname{PE}_{(pos,2i+1)}
=
\cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

## Encoder Block Walkthrough

An encoder layer starts from token states:

$$
H^{(l-1)} \in \mathbb{R}^{T_x \times d_{\text{model}}}
$$

Self-attention mixes source tokens:

$$
A^{(l)}
=
\operatorname{MultiHead}(H^{(l-1)}, H^{(l-1)}, H^{(l-1)})
$$

The paper uses residual connections and layer normalization around each sublayer:

$$
\tilde{H}^{(l)}
=
\operatorname{LayerNorm}(H^{(l-1)} + A^{(l)})
$$

Then a position-wise feed-forward network is applied independently at each token:

$$
\operatorname{FFN}(h)
=
\max(0, hW_1 + b_1)W_2 + b_2
$$

$$
H^{(l)}
=
\operatorname{LayerNorm}(\tilde{H}^{(l)} + \operatorname{FFN}(\tilde{H}^{(l)}))
$$

The important separation is:

| Sublayer | Mixes tokens? | Mixes channels? |
| --- | --- | --- |
| self-attention | yes | yes, through projections |
| feed-forward network | no | yes |
| residual/normalization | no | stabilizes representation scale |

This is why Transformer blocks are often described as alternating token mixing and channel mixing.

## Decoder Block Walkthrough

The decoder has three sublayers.

| Sublayer | Query | Key/Value | Constraint |
| --- | --- | --- | --- |
| masked self-attention | target prefix states | target prefix states | cannot attend to future target positions |
| cross-attention | target states | encoder source states | can attend to all source positions |
| feed-forward network | each target position | same target position | position-wise |

The causal mask changes decoder self-attention:

$$
\operatorname{MaskedAttention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

where:

$$
M_{ij}
=
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

Cross-attention then lets each target position query the encoded source sequence:

$$
\operatorname{CrossAttn}(H_y, H_x)
=
\operatorname{MultiHead}(Q=H_y, K=H_x, V=H_x)
$$

This encoder-decoder structure is different from later [[concepts/architectures/encoder-only-transformer|encoder-only]] and [[concepts/architectures/decoder-only-transformer|decoder-only]] Transformer backbones.

## Complexity and Inductive Bias

The paper compares self-attention, recurrent layers, and convolutional layers by path length and computational structure.

For sequence length $n$ and hidden dimension $d$, dense self-attention has:

$$
O(n^2 d)
$$

time/multiply cost for attention interactions, while recurrent layers have sequential path length:

$$
O(n)
$$

and self-attention has maximum token-token path length:

$$
O(1)
$$

The tradeoff is direct global interaction versus quadratic sequence-length cost.

| Architecture | Main Bias | Parallel Over Positions | Long-Range Path |
| --- | --- | --- | --- |
| RNN | temporal recurrence | no | long |
| CNN | locality and translation sharing | yes | grows with depth/dilation |
| Transformer | content-based global addressing | yes | short |

This is the core reason the paper became more than a translation result: it changed the default assumption about how sequence positions should communicate.

## Training Recipe

The architecture is not the only thing that matters. The reported results also depend on the training recipe.

| Ingredient | Role |
| --- | --- |
| residual dropout | regularizes sublayers and embeddings |
| label smoothing | avoids overconfident target distributions |
| Adam with warmup schedule | stabilizes early optimization |
| byte-pair encoding | controls vocabulary and open-word handling |
| beam search | affects reported translation quality |

The learning-rate schedule is part of the paper's practical recipe:

$$
\operatorname{lrate}
=
d_{\text{model}}^{-0.5}
\cdot
\min
\left(
\operatorname{step}^{-0.5},
\operatorname{step}\cdot \operatorname{warmup}^{-1.5}
\right)
$$

For this wiki, architecture conclusions should not ignore these recipe choices. A weaker training setup may make the same block look worse; a stronger setup may improve baselines too.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Attention-only encoder-decoder can outperform strong translation baselines | WMT 2014 English-German and English-French BLEU results | claim is under machine translation protocol |
| Transformer trains more parallelly than recurrent sequence models | architecture removes sequential recurrence across token positions | wall-clock depends on hardware and implementation |
| Multi-head attention and feed-forward blocks are useful components | ablations over heads, key/value dimensions, model size, and positional encoding | ablations are mostly within translation setup |
| Model transfers beyond translation | English constituency parsing experiments | not a broad proof of universal transfer |

## Ablation Reading

The most useful ablations are about which architectural pieces are necessary.

| Ablation Axis | What it tests | Reading |
| --- | --- | --- |
| number of heads | whether multiple attention subspaces help | too few heads can bottleneck relation types; too many reduce per-head dimension |
| key/value dimension | capacity and attention sharpness | smaller dimensions can reduce compute but lose quality |
| model size | depth/width scaling | improvement is not solely from attention form |
| positional encoding | fixed sinusoidal vs learned | position injection matters, but the paper does not settle all later position methods |
| feed-forward dimension | channel-mixing capacity | attention alone is not the whole architecture |

The paper supports the Transformer block as a package: attention, FFN, residuals, normalization, position encoding, optimization, and enough scale.

## Benchmark Card

| Field | WMT translation setting |
| --- | --- |
| Task | sequence-to-sequence machine translation |
| Input/output unit | source sentence to target sentence |
| Main metric | BLEU |
| Main comparison | recurrent/convolutional sequence transduction baselines and prior state of the art |
| Generalization claim | translation quality under the benchmark's train/test setup |
| Not directly tested | modern LLM pretraining, retrieval, tool use, multimodal reasoning, protein modeling |

## What Later Models Kept and Changed

| Kept | Changed Later |
| --- | --- |
| scaled dot-product attention | sparse, local, linear, grouped-query, and flash attention variants |
| multi-head projection pattern | multi-query and grouped-query attention for inference efficiency |
| residual block structure | pre-norm and other normalization placement choices |
| feed-forward sublayer | GELU, SwiGLU, gated FFNs, mixture-of-experts FFNs |
| positional signal | relative position, rotary embeddings, ALiBi, learned position schemes |
| encoder-decoder idea | encoder-only BERT and decoder-only GPT-style models |

This matters because "Transformer" now names a family. The 2017 paper is the root, but many modern systems are not exact copies of the original encoder-decoder stack.

## Implementation Notes

- Attention masks are part of the architecture contract; encoder padding masks, decoder causal masks, and cross-attention masks must be handled separately.
- Tensor shapes should be explicit: batches, heads, sequence length, and head dimension are easy sources of silent bugs.
- Positional encoding must be added or otherwise represented; self-attention alone is permutation-equivariant over token order.
- LayerNorm placement changes training stability, especially in deeper stacks.
- Teacher forcing during training differs from autoregressive decoding at inference.
- BLEU improvements can be affected by tokenization, beam size, checkpoint averaging, and evaluation scripts.

## Limitations

- The paper's evidence is centered on translation and parsing, not today's broad LLM setting.
- BLEU is useful for translation comparison but does not measure all downstream language-model behavior.
- The architecture changes compute structure as well as modeling inductive bias, so architecture, hardware efficiency, and implementation are intertwined.
- Long-context behavior beyond trained/evaluated lengths requires separate evidence.
- Later Transformer variants changed normalization placement, positional encoding, activation, scaling, objective, data, and training infrastructure.
- Dense attention has quadratic memory and compute in sequence length.
- The original architecture does not directly address retrieval, tool use, factual grounding, or alignment.
- Many later "Transformer" claims depend more on data and scale than on the original block alone.

## Why It Matters

This paper is the anchor for modern Transformer-based systems. For this wiki, it should be treated as:

- an [[papers/architectures/index|architecture paper]];
- a prerequisite for [[concepts/architectures/transformer|Transformer]];
- a bridge from [[concepts/architectures/attention|Attention]] to [[concepts/architectures/encoder-decoder|encoder-decoder architectures]];
- a historical foundation for [[agents/index|Agents]], while not itself being a modern LLM paper.

The reusable architecture pattern is:

$$
\text{token embedding}
+
\text{position signal}
\rightarrow
\left[
\text{attention token mixing}
+
\text{FFN channel mixing}
\right]^L
\rightarrow
\text{task head or decoder distribution}
$$

That pattern later branches into [[papers/architectures/bert|BERT]], [[papers/architectures/gpt-2|GPT-2]], [[papers/architectures/vision-transformer|Vision Transformer]], [[papers/architectures/swin-transformer|Swin Transformer]], and many biological sequence or structure models.

## Connections

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[ai/architectures|AI architectures]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/vision-transformer|Vision Transformer]]
- [[papers/architectures/index|Architecture papers]]
