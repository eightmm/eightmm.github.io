---
title: BERT
aliases:
  - papers/bert
  - papers/pre-training-of-deep-bidirectional-transformers
tags:
  - papers
  - architectures
  - transformer
  - language-model
---

# BERT

> The paper made the encoder-only Transformer a reusable bidirectional language representation backbone.

## Metadata

| Field | Value |
| --- | --- |
| Paper | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding |
| Authors | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova |
| Year | 2019 |
| Venue | NAACL-HLT 2019 |
| arXiv | [1810.04805](https://arxiv.org/abs/1810.04805) |
| ACL Anthology | [N19-1423](https://aclanthology.org/N19-1423/) |
| Status | verified |

## Question

Before BERT, many language models were left-to-right or shallowly bidirectional. The question was whether a deep Transformer encoder could learn reusable bidirectional token representations from unlabeled text and then adapt to many supervised NLP tasks.

## Main Claim

BERT pre-trains a deep bidirectional Transformer encoder with masked language modeling and next-sentence prediction, then fine-tunes the same backbone with small task heads.

Masked language modeling objective:

$$
\mathcal{L}_{\text{MLM}}
=
-
\sum_{i \in M}
\log p_\theta(x_i \mid x_{\setminus M})
$$

where $M$ is the set of masked token positions.

## Method

| Component | Role |
| --- | --- |
| Transformer encoder | bidirectional token mixing |
| `[CLS]` token | sequence-level representation |
| masked language modeling | forces contextual token reconstruction |
| next-sentence prediction | trains a sentence-pair signal in the original setup |
| fine-tuning head | adapts shared backbone to downstream tasks |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Bidirectional encoder pre-training improves NLP tasks | GLUE, MultiNLI, SQuAD, and other benchmark gains | objective and data scale are coupled with architecture |
| One backbone can support many tasks | fine-tuning with small task-specific heads | task formatting still matters |
| Deep bidirectional context is useful | comparison against prior contextual representations | later work revised NSP and pre-training recipes |

## Limitations

- BERT is not an autoregressive generator; it is mainly an encoder representation model.
- The paper mixes architecture, pre-training objective, data, and fine-tuning recipe.
- Maximum sequence length and quadratic attention limit long-context use.
- Later encoder models changed data, objectives, scaling, and training details.

## Why It Matters

BERT is the canonical paper for encoder-only Transformers as reusable language representation backbones.

## Connections

- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/llm/language-model|Language model]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
