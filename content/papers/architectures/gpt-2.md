---
title: Language Models are Unsupervised Multitask Learners
aliases:
  - papers/gpt-2
  - papers/language-models-are-unsupervised-multitask-learners
tags:
  - papers
  - architectures
  - transformer
  - language-model
---

# Language Models are Unsupervised Multitask Learners

> The paper made the decoder-only Transformer language model a strong zero-shot task interface at scale.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Language Models are Unsupervised Multitask Learners |
| Authors | Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. |
| Year | 2019 |
| Venue | OpenAI technical report |
| PDF | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| Code | [openai/gpt-2](https://github.com/openai/gpt-2) |
| Status | verified |

## Question

Most NLP systems were trained or fine-tuned for specific supervised tasks. The question was whether a sufficiently large autoregressive language model trained on broad web text could perform many tasks from natural-language context alone.

## Main Claim

A decoder-only Transformer trained with next-token prediction can acquire broad zero-shot task behavior when model capacity and data diversity increase.

Autoregressive objective:

$$
\mathcal{L}_{\text{LM}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t})
$$

The architecture interface is:

$$
\text{prompt tokens}
\rightarrow
\text{causal Transformer}
\rightarrow
\text{next-token distribution}
$$

## Method

| Component | Role |
| --- | --- |
| decoder-only Transformer | causal token mixing |
| byte-pair encoding | open-vocabulary tokenization |
| WebText | broad next-token training data |
| prompt conditioning | expresses tasks as text context |
| scale sweep | tests whether capacity improves transfer |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Larger language models improve zero-shot transfer | task benchmarks under no fine-tuning | prompting and evaluation protocols were early |
| Next-token prediction can encode task behavior | question answering, summarization, translation, and reading comprehension tests | data contamination and benchmark framing require care |
| Decoder-only Transformers scale as general sequence models | model-size comparisons | architecture, data, and compute are inseparable |

## Limitations

- GPT-2 is a scaling and data paper as much as an architecture note.
- Zero-shot evaluation is sensitive to prompt format and benchmark leakage.
- Causal attention cannot condition on future tokens directly.
- The original report predates modern instruction tuning, RLHF, tool use, and chat interfaces.

## Why It Matters

GPT-2 is the practical bridge from Transformer language modeling to prompt-based general-purpose LLM interfaces.

## Connections

- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/llm/language-model|Language model]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/index|Architecture papers]]
