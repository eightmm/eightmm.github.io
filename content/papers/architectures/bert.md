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

More concretely, the paper asks whether a single pre-trained encoder can replace many task-specific NLP architectures. The architectural bet is that deep bidirectional self-attention is a better default representation engine than a stack of task-specific feature extractors.

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

Next-sentence prediction in the original paper is a binary classification objective:

$$
\mathcal{L}_{\text{NSP}}
=
-
\log p_\theta(y_{\text{is-next}} \mid x_A, x_B)
$$

The combined pre-training loss is:

$$
\mathcal{L}
=
\mathcal{L}_{\text{MLM}}
+
\mathcal{L}_{\text{NSP}}
$$

The claim should be read narrowly: BERT shows that encoder-only Transformer pre-training plus fine-tuning is powerful for language understanding benchmarks. It does not claim that the encoder-only form is the best architecture for generation.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | tokenized text sequence or text pair |
| Special tokens | `[CLS]` at the start, `[SEP]` between or after segments |
| Token representation | token embedding + segment embedding + positional embedding |
| Backbone | stacked bidirectional Transformer encoder blocks |
| Context direction | every non-masked token can attend to left and right context |
| Sequence output | contextual vector for each token |
| Sequence-level output | `[CLS]` vector for classification-style heads |

The input embedding for position $i$ can be written as:

$$
e_i
=
E_{\text{token}}(x_i)
+
E_{\text{segment}}(s_i)
+
E_{\text{position}}(i)
$$

The encoder maps token embeddings into contextual states:

$$
H
=
\operatorname{TransformerEncoder}_\theta(E)
$$

where:

$$
H \in \mathbb{R}^{T \times d}
$$

with sequence length $T$ and hidden dimension $d$.

## Encoder Block Walkthrough

BERT inherits the Transformer encoder block from [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], but uses it as a pre-trained representation stack rather than a translation encoder.

One block can be summarized as:

$$
Z^{(l)}
=
\operatorname{SelfAttn}(H^{(l-1)})
$$

$$
\tilde{H}^{(l)}
=
\operatorname{LayerNorm}(H^{(l-1)} + Z^{(l)})
$$

$$
F^{(l)}
=
\operatorname{FFN}(\tilde{H}^{(l)})
$$

$$
H^{(l)}
=
\operatorname{LayerNorm}(\tilde{H}^{(l)} + F^{(l)})
$$

The exact normalization placement follows the original post-norm Transformer style. Later Transformer stacks often use pre-norm variants for more stable scaling.

The key architectural difference from a decoder-only language model is the attention mask:

| Model Type | Attention Pattern | Natural Use |
| --- | --- | --- |
| encoder-only Transformer | bidirectional attention | representation, classification, extraction |
| decoder-only Transformer | causal attention | generation, prompting, next-token prediction |
| encoder-decoder Transformer | bidirectional source encoder + causal target decoder | sequence-to-sequence transduction |

## Method

| Component | Role |
| --- | --- |
| Transformer encoder | bidirectional token mixing |
| `[CLS]` token | sequence-level representation |
| masked language modeling | forces contextual token reconstruction |
| next-sentence prediction | trains a sentence-pair signal in the original setup |
| fine-tuning head | adapts shared backbone to downstream tasks |

## Pre-training Pipeline

BERT pre-training has three important design choices.

| Choice | Meaning | Why it matters |
| --- | --- | --- |
| masked token prediction | predict selected tokens from context | permits bidirectional conditioning without trivial copying |
| sentence-pair formatting | pack one or two segments with segment embeddings | supports entailment, QA, and sentence-pair tasks |
| task-agnostic backbone | keep the encoder shared across tasks | makes the paper a pre-trained architecture paper, not only an NLP benchmark paper |

The masking recipe is easy to misread. The model does not simply replace every target token with `[MASK]` at fine-tuning time. A subset of tokens is selected for prediction; selected positions are mostly replaced by `[MASK]`, sometimes left unchanged, and sometimes replaced by a random token. This reduces mismatch between pre-training and fine-tuning because `[MASK]` is not present in downstream input.

## Fine-tuning Routes

The paper's practical contribution is the simple route from one encoder to many downstream heads.

| Task Type | Output Used | Head |
| --- | --- | --- |
| sentence classification | `[CLS]` hidden state | linear classifier |
| sentence-pair classification | `[CLS]` after packed pair input | linear classifier |
| token classification | per-token hidden states | token-level classifier |
| extractive question answering | per-token hidden states | start/end span classifiers |

For classification:

$$
p(y \mid x)
=
\operatorname{softmax}(W h_{\text{[CLS]}} + b)
$$

For extractive QA:

$$
p_{\text{start}}(i \mid x)
=
\operatorname{softmax}_i(w_s^\top h_i)
$$

$$
p_{\text{end}}(i \mid x)
=
\operatorname{softmax}_i(w_e^\top h_i)
$$

This is why BERT became a backbone: most task-specific architecture is reduced to a small prediction head.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Bidirectional encoder pre-training improves NLP tasks | GLUE, MultiNLI, SQuAD, and other benchmark gains | objective and data scale are coupled with architecture |
| One backbone can support many tasks | fine-tuning with small task-specific heads | task formatting still matters |
| Deep bidirectional context is useful | comparison against prior contextual representations | later work revised NSP and pre-training recipes |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | language understanding |
| Input/output unit | text or text pair to class label, token label, or answer span |
| Main benchmarks | GLUE, MultiNLI, SQuAD, named entity recognition, and related NLP tasks |
| Main comparison | prior contextual representation models and task-specific systems |
| Main metric types | accuracy, F1, exact match, task-specific benchmark scores |
| Not directly tested | open-ended generation, tool use, retrieval-augmented reasoning, multimodal modeling |

## Ablation Reading

The most useful ablations are not just "BERT is better." They answer which ingredients are carrying the result.

| Ablation Axis | What it tests | Reading |
| --- | --- | --- |
| bidirectional vs left-to-right context | whether full-context encoder attention matters | supports the encoder-only representation claim |
| MLM and NSP objectives | whether each pre-training signal helps | later work weakened the case for NSP as a universal requirement |
| model size | whether depth/width improves transfer | supports scaling, but data and optimization also change effective capacity |
| pre-training duration/data | whether representation quality is data-dependent | architecture should not be isolated from corpus and compute |

The main architectural takeaway is the bidirectional encoder stack. The MLM/NSP recipe is important historically, but later encoder models changed the exact objective while keeping the encoder-only backbone idea.

## What To Reuse

For this wiki, BERT should be reused as a pattern, not only as a named model.

| Reusable Pattern | Where it appears later |
| --- | --- |
| pre-train a generic encoder, fine-tune small heads | NLP classification, retrieval, protein language models |
| mask parts of the input and reconstruct them | [[concepts/learning/masked-modeling|masked modeling]], MAE-style vision models, protein sequence pre-training |
| encode pairs with segment/context structure | sentence-pair tasks, cross-encoder rerankers |
| use pooled sequence state for classification | representation evaluation and benchmark probing |

## Implementation Notes

- The `[CLS]` vector is not automatically a semantic sentence embedding unless trained/evaluated for that use.
- Tokenization affects vocabulary coverage, span boundaries, and downstream error analysis.
- Fine-tuning is sensitive to learning rate, batch size, warmup, sequence length, and random seed.
- Long documents require truncation, sliding windows, retrieval, or long-context encoder variants.
- For retrieval systems, bi-encoder and cross-encoder BERT variants have different latency/accuracy tradeoffs.

## Limitations

- BERT is not an autoregressive generator; it is mainly an encoder representation model.
- The paper mixes architecture, pre-training objective, data, and fine-tuning recipe.
- Maximum sequence length and quadratic attention limit long-context use.
- Later encoder models changed data, objectives, scaling, and training details.
- NSP should not be treated as mandatory for all encoder pre-training; the later literature changed this recipe.
- Benchmark gains do not by themselves prove robust out-of-distribution language understanding.

## Why It Matters

BERT is the canonical paper for encoder-only Transformers as reusable language representation backbones.

It also explains a recurring pattern in AI architecture papers:

$$
\text{architecture}
+
\text{self-supervised pre-training}
+
\text{simple adaptation head}
\rightarrow
\text{general-purpose backbone}
$$

That pattern later appears in vision, speech, protein language modeling, and multimodal representation learning.

## Connections

- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/llm/language-model|Language model]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/deberta|DeBERTa]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/index|Architecture papers]]
