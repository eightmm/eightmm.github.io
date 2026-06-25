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

Here $X$ is the token matrix, $M$ is a padding or causal mask, $H$ is the number of heads, and $\sigma$ is the feed-forward nonlinearity. Decoder-only Transformers use a causal $M$; encoder-style models usually use bidirectional attention with only padding masks. The block combines [[concepts/architectures/embedding|embeddings]], [[concepts/architectures/linear-layer|linear layers]], [[concepts/architectures/softmax|softmax]], [[concepts/architectures/normalization|normalization]], and [[concepts/architectures/residual-connection|residual connections]].

## Key Ideas

- Self-attention lets each token mix information from other tokens according to learned relevance.
- Feed-forward blocks, normalization, residual connections, and positional encodings are part of the core pattern.
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]], [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]], and [[concepts/architectures/encoder-decoder|encoder-decoder]] variants serve different training and inference workflows.
- Causal masking supports autoregressive generation; bidirectional attention supports representation learning.
- Transformers can process text, protein sequences, molecular strings, retrieved context, and multimodal tokens when inputs are tokenized.

## Practical Checks

- Identify the variant: encoder-only, decoder-only, or [[concepts/architectures/encoder-decoder|encoder-decoder]].
- Check context length, attention pattern, positional encoding, and masking.
- Track what each token represents: wordpiece, residue, atom token, graph node, image patch, tool call, or retrieved chunk.
- For scientific tasks, separate model architecture from data split, leakage, calibration, and downstream evaluation.

## Related

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[entities/protein|Protein]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[agents/coding-agents|Coding agents]]
