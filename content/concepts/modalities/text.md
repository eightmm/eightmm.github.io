---
title: Text
tags:
  - modalities
  - text
  - language-models
---

# Text

Text is a discrete symbolic sequence. In neural models, raw text is usually tokenized, embedded, and processed by a sequence architecture such as a [[concepts/architectures/transformer|Transformer]], [[concepts/architectures/rnn|RNN]], or [[concepts/architectures/state-space-model|state-space model]].

Tokenization maps a string to discrete IDs:

$$
t_{1:L} = \operatorname{Tokenizer}(x_{\text{text}})
$$

Embedding maps tokens to vectors:

$$
X = [E_{t_1}+p_1,\ldots,E_{t_L}+p_L]^\top
$$

where $E_{t_i}$ is the token embedding and $p_i$ is a positional or structural encoding.

## Key Ideas

- Token granularity can be byte, character, subword, word, sentence, or document chunk.
- Context length controls how much text can be jointly conditioned on.
- Formatting, markup, control tokens, and prompt boundaries are part of the input distribution.
- Text can be an instruction, evidence, code, metadata, or untrusted user-provided data.

## Checks

- What does the tokenizer split or merge?
- Is important context truncated?
- Are prompts, retrieved documents, and tool observations separated clearly?
- Does evaluation test language understanding, memorization, formatting, or retrieval?

## Related

- [[concepts/modalities/sequence|Sequence]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
