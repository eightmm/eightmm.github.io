---
title: Transformer
tags:
  - architectures
  - transformer
  - attention
---

# Transformer

Transformers use attention to model relationships across tokens. They are common in language models, protein sequence models, molecular sequence models, and agent systems.

## Key Ideas

- Self-attention lets each token mix information from other tokens according to learned relevance.
- Feed-forward blocks, normalization, residual connections, and positional encodings are part of the core pattern.
- Encoder-only, decoder-only, and encoder-decoder variants serve different training and inference workflows.
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
