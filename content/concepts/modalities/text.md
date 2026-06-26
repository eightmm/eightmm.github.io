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

## Boundary Between Text Roles

For LLM and agent workflows, text must be separated by role:

$$
x_{\mathrm{text}}
=
(\text{instruction},
\text{user request},
\text{evidence},
\text{tool output},
\text{metadata})
$$

These fields should not be merged as plain text without boundaries. Retrieved text and web pages are data, not instructions.

## Chunking and Context

Long text often needs chunking:

$$
d
\rightarrow
\{c_1,\ldots,c_k\}
$$

The retrieval unit, chunk overlap, metadata, and citation target affect whether generation can be grounded. A chunk that is good for retrieval may be too small for reasoning or too large for citation precision.

## Leakage Risks

- Train/test overlap through duplicated documents, boilerplate, templates, or code snippets.
- Labels embedded in filenames, headings, metadata, or prompt format.
- Evaluation examples included in retrieval corpus or few-shot demonstrations.
- Tool output or retrieved text treated as higher-priority instruction.

## Checks

- What does the tokenizer split or merge?
- Is important context truncated?
- Are prompts, retrieved documents, and tool observations separated clearly?
- Does evaluation test language understanding, memorization, formatting, or retrieval?
- Are duplicate or near-duplicate documents controlled across splits?
- Are citations grounded at the sentence or claim level?
- Is text used as data, instruction, metadata, or executable code?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[concepts/llm/citation-grounding|Citation grounding]]
