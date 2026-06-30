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
| Korean longform | [[posts/essays/attention-is-all-you-need-transformer-review|Attention Is All You Need를 지금 다시 읽는 법]] |

## Question

Before this paper, strong sequence transduction systems were usually built around recurrent or convolutional encoder-decoder models, often with attention added on top. The question was whether recurrence and convolution were necessary for high-quality sequence modeling, or whether attention alone could do the sequence mixing.

## Main Claim

The paper's central claim is that a sequence transduction model can be built entirely from attention and feed-forward blocks, without recurrence or convolution, while improving translation quality and training parallelism on the reported machine translation benchmarks.

Narrowed claim:

$$
\text{Transformer}
\Rightarrow
\text{strong WMT translation results under the paper's training and evaluation protocol}
$$

This is not the same as proving that recurrence or convolution are never useful. It shows that attention-only sequence modeling is a strong architecture class under the tested settings.

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

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Attention-only encoder-decoder can outperform strong translation baselines | WMT 2014 English-German and English-French BLEU results | claim is under machine translation protocol |
| Transformer trains more parallelly than recurrent sequence models | architecture removes sequential recurrence across token positions | wall-clock depends on hardware and implementation |
| Multi-head attention and feed-forward blocks are useful components | ablations over heads, key/value dimensions, model size, and positional encoding | ablations are mostly within translation setup |
| Model transfers beyond translation | English constituency parsing experiments | not a broad proof of universal transfer |

## Benchmark Card

| Field | WMT translation setting |
| --- | --- |
| Task | sequence-to-sequence machine translation |
| Input/output unit | source sentence to target sentence |
| Main metric | BLEU |
| Main comparison | recurrent/convolutional sequence transduction baselines and prior state of the art |
| Generalization claim | translation quality under the benchmark's train/test setup |
| Not directly tested | modern LLM pretraining, retrieval, tool use, multimodal reasoning, protein modeling |

## Limitations

- The paper's evidence is centered on translation and parsing, not today's broad LLM setting.
- BLEU is useful for translation comparison but does not measure all downstream language-model behavior.
- The architecture changes compute structure as well as modeling inductive bias, so architecture, hardware efficiency, and implementation are intertwined.
- Long-context behavior beyond trained/evaluated lengths requires separate evidence.
- Later Transformer variants changed normalization placement, positional encoding, activation, scaling, objective, data, and training infrastructure.

## Why It Matters

This paper is the anchor for modern Transformer-based systems. For this wiki, it should be treated as:

- an [[papers/architectures/index|architecture paper]];
- a prerequisite for [[concepts/architectures/transformer|Transformer]];
- a bridge from [[concepts/architectures/attention|Attention]] to [[concepts/architectures/encoder-decoder|encoder-decoder architectures]];
- a historical foundation for [[papers/llm/index|LLM papers]], while not itself being a modern LLM paper.

## Connections

- [[posts/essays/attention-is-all-you-need-transformer-review|Attention Is All You Need를 지금 다시 읽는 법]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[ai/architectures|AI architectures]]
- [[papers/architectures/index|Architecture papers]]
