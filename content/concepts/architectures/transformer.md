---
title: Transformer
tags:
  - architectures
  - transformer
  - attention
---

# Transformer

Transformers use attention to model relationships across tokens. They are common in language models, protein sequence models, molecular sequence models, and agent systems.

A pre-norm Transformer block with multi-head self-attention is:

$$
\tilde{X} = \operatorname{LN}(X)
$$

$$
Q_h = \tilde{X}W_Q^{(h)},\qquad
K_h = \tilde{X}W_K^{(h)},\qquad
V_h = \tilde{X}W_V^{(h)}
$$

$$
\operatorname{head}_h
= \operatorname{softmax}
\left(
\frac{Q_hK_h^\top}{\sqrt{d_k}} + M
\right)V_h
$$

$$
\operatorname{MHA}(\tilde{X})
= \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
$$

$$
X' = X + \operatorname{MHA}(\tilde{X})
$$

$$
\operatorname{FFN}(Z)
= \sigma(ZW_1+b_1)W_2+b_2
$$

$$
X_{\mathrm{out}}
= X' + \operatorname{FFN}(\operatorname{LN}(X'))
$$

Here $X$ is the token matrix, $M$ is a padding or causal mask, $H$ is the number of heads, and $\sigma$ is the feed-forward nonlinearity. Decoder-only Transformers use a causal $M$; encoder-style models usually use bidirectional attention with only padding masks. The block combines [[concepts/architectures/tokenization|tokenization]], [[concepts/architectures/embedding|embeddings]], [[concepts/architectures/linear-layer|linear layers]], [[concepts/architectures/softmax|softmax]], [[concepts/architectures/normalization|normalization]], [[concepts/architectures/residual-connection|residual connections]], and a token-wise [[concepts/architectures/feed-forward-network|feed-forward network]].

## Shape Contract

For a batch of token embeddings:

$$
X\in\mathbb{R}^{B\times T\times d_{\mathrm{model}}}
$$

each head uses projections:

$$
W_Q^{(h)},W_K^{(h)},W_V^{(h)}
\in
\mathbb{R}^{d_{\mathrm{model}}\times d_k}
$$

and produces:

$$
\operatorname{head}_h
\in
\mathbb{R}^{B\times T_q\times d_v}
$$

The attention score tensor has shape:

$$
S_h
=
\frac{Q_hK_h^\top}{\sqrt{d_k}}
\in
\mathbb{R}^{B\times T_q\times T_k}
$$

For self-attention, $T_q=T_k=T$. For cross-attention, query tokens and key/value tokens may come from different sources.

## Mask Semantics

The mask $M$ defines what information is available:

| Mask | Allows | Typical use |
| --- | --- | --- |
| padding mask | ignore nonexistent tokens | variable-length batches |
| causal mask | attend only to previous tokens | autoregressive decoding |
| bidirectional mask | attend to all valid tokens | representation learning |
| block/local mask | attend to restricted windows | long-context efficiency |
| cross mask | connect query tokens to external memory | retrieval or encoder-decoder models |

Mask choice is part of the model claim. A representation model, a generator, and a retrieved-context agent can all use Transformers while having different information boundaries.

## Complexity

Full self-attention scales quadratically in sequence length:

$$
\operatorname{cost}_{\mathrm{attn}}
=
O(BHT^2d_k)
$$

and attention logits require:

$$
O(BHT^2)
$$

memory before optimization. This is why long-context systems use caching, local attention, sparse attention, state-space models, retrieval, or chunking.

## Key Ideas

- Self-attention lets each token mix information from other tokens according to learned relevance.
- [[concepts/architectures/feed-forward-network|Feed-forward networks]], normalization, residual connections, and positional encodings are part of the core pattern.
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]], [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]], and [[concepts/architectures/encoder-decoder|encoder-decoder]] variants serve different training and inference workflows.
- Causal masking supports autoregressive generation; bidirectional attention supports representation learning.
- Transformers can process text, protein sequences, molecular strings, retrieved context, and multimodal tokens when inputs are tokenized.

## Practical Checks

- Identify the variant: encoder-only, decoder-only, or [[concepts/architectures/encoder-decoder|encoder-decoder]].
- Check context length, attention pattern, positional encoding, and masking.
- Track what each token represents: wordpiece, residue, atom token, graph node, image patch, tool call, or retrieved chunk.
- For scientific tasks, separate model architecture from data split, leakage, calibration, and downstream evaluation.
- Check whether reported context length is training length, inference length, or extrapolated length.
- Check whether memory optimization changes exact attention, approximation, or only implementation.

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/weight-initialization|Weight initialization]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[entities/protein|Protein]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[agents/workflows/coding-agents|Coding agents]]
