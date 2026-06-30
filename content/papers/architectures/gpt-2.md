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

The architectural question is whether the decoder side of the Transformer can become a general interface: provide a text prefix, let the model continue it, and express tasks through the prefix rather than through a new supervised head.

This moves the center of gravity from:

$$
\text{task-specific model}
\rightarrow
\text{task-specific dataset}
\rightarrow
\text{task-specific head}
$$

to:

$$
\text{general language model}
\rightarrow
\text{text prompt}
\rightarrow
\text{next-token continuation}
$$

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

This should be read as a transfer claim, not a proof that supervised task learning is unnecessary. The paper shows that next-token pre-training plus scale can expose many task behaviors through prompting.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | token prefix |
| Output | probability distribution over the next token |
| Backbone | decoder-only Transformer |
| Attention pattern | causal self-attention |
| Training objective | next-token prediction |
| Adaptation interface | prompt/context formatting |
| Natural use | generation, completion, zero-shot/few-shot task framing |

For token sequence $x_{1:T}$, the model factorizes the sequence probability as:

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$

The hidden states are produced by a causal Transformer:

$$
H
=
\operatorname{CausalTransformer}_\theta(E(x_{1:T}))
$$

and next-token probabilities are:

$$
p_\theta(x_{t+1} \mid x_{\le t})
=
\operatorname{softmax}(W_o h_t)
$$

The key restriction is that $h_t$ cannot attend to future tokens.

## Decoder-Only Block

GPT-2 uses the decoder-only branch of the Transformer family. Compared with [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], it removes the encoder and cross-attention path and keeps only causal self-attention plus feed-forward blocks.

One layer can be summarized as:

$$
A^{(l)}
=
\operatorname{MaskedSelfAttn}(H^{(l-1)})
$$

$$
\tilde{H}^{(l)}
=
\operatorname{LayerNorm}(H^{(l-1)} + A^{(l)})
$$

$$
H^{(l)}
=
\operatorname{LayerNorm}(\tilde{H}^{(l)} + \operatorname{FFN}(\tilde{H}^{(l)}))
$$

The causal attention mask is:

$$
M_{ij}
=
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

so attention is:

$$
\operatorname{MaskedAttention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

The decoder-only simplification is what makes the same architecture useful for open-ended continuation.

## Method

| Component | Role |
| --- | --- |
| decoder-only Transformer | causal token mixing |
| byte-pair encoding | open-vocabulary tokenization |
| WebText | broad next-token training data |
| prompt conditioning | expresses tasks as text context |
| scale sweep | tests whether capacity improves transfer |

## Prompt Interface

GPT-2 is important because it treats tasks as text continuation problems.

| Task | Prompt-like framing | Output behavior |
| --- | --- | --- |
| translation | source text plus translation cue | target-language continuation |
| summarization | document plus summary cue | summary continuation |
| question answering | context/question text | answer continuation |
| reading comprehension | passage and query | completion with answer span or answer text |

This is the predecessor of later prompt engineering and in-context learning. The paper does not yet have modern chat formatting, tool calling, system messages, or RLHF, but it makes the interface shift visible.

The general template is:

$$
\text{task description or examples}
+
\text{input}
\rightarrow
\text{desired output as continuation}
$$

## Scaling Claim

The paper compares models of different sizes and argues that zero-shot task performance improves with scale.

The implicit claim is:

$$
\text{larger decoder-only LM}
+
\text{broader data}
\Rightarrow
\text{better zero-shot transfer}
$$

For this wiki, keep the claim narrow:

- the evidence is from the paper's selected zero-shot benchmarks;
- improvements are tied to model size, data, tokenization, and evaluation framing;
- zero-shot behavior is not the same as reliable instruction following;
- a scaling trend is not a proof of robust reasoning.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Larger language models improve zero-shot transfer | task benchmarks under no fine-tuning | prompting and evaluation protocols were early |
| Next-token prediction can encode task behavior | question answering, summarization, translation, and reading comprehension tests | data contamination and benchmark framing require care |
| Decoder-only Transformers scale as general sequence models | model-size comparisons | architecture, data, and compute are inseparable |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | zero-shot NLP transfer through language modeling |
| Input/output unit | prompt/context text to continuation text |
| Main model family | decoder-only Transformer |
| Main objective | autoregressive next-token prediction |
| Main comparison | smaller GPT-style models and task-specific systems |
| Main metrics | task-specific accuracy, F1, perplexity-like or benchmark-specific scores |
| Not directly tested | instruction tuning, RLHF, tool use, retrieval-grounded generation, agent loops |

## Ablation Reading

GPT-2 is less of a clean architecture ablation paper than ResNet or the original Transformer. Its strongest reading is about the interaction between architecture, scale, and data.

| Axis | What it tests | Reading |
| --- | --- | --- |
| model size | whether capacity improves transfer | supports scaling, not architecture alone |
| broad web pre-training | whether diverse text creates task exposure | raises contamination and distribution questions |
| zero-shot prompting | whether tasks can be expressed as continuation | sensitive to prompt format |
| next-token objective | whether one objective supports many behaviors | does not isolate supervised fine-tuning alternatives |

The paper should not be cited as proving that decoder-only Transformers are universally superior. It is evidence that decoder-only language modeling becomes a powerful general interface when scaled.

## Relation to BERT

| Paper | Backbone | Objective | Natural Use |
| --- | --- | --- | --- |
| [BERT](/papers/architectures/bert) | encoder-only Transformer | masked token reconstruction | representation and understanding tasks |
| GPT-2 | decoder-only Transformer | next-token prediction | generation and prompt-conditioned transfer |

BERT asks: can bidirectional pre-training create reusable representations?

GPT-2 asks: can autoregressive pre-training create reusable behavior through text continuation?

Both depend on [[papers/architectures/attention-is-all-you-need|the Transformer]], but they split the architecture family into two important branches.

## From GPT-2 to Agents

GPT-2 is not an agent paper, but it is part of the path toward LLM agents.

| Capability Layer | GPT-2 Contribution | Missing Piece |
| --- | --- | --- |
| text continuation | strong decoder-only LM interface | instruction alignment |
| prompt conditioning | tasks expressed in context | robust prompt following |
| broad task behavior | zero-shot benchmark transfer | tool use and external state |
| generated action text | language can represent actions | execution, verification, memory |

Modern agents add tool contracts, memory, environment feedback, planning loops, and verification. GPT-2 contributes the base interface: a language model can condition on context and produce task-relevant continuations.

## Implementation Notes

- Causal masks must be correct; a leaked future token invalidates autoregressive training.
- Tokenization affects context length, rare words, code-like text, and benchmark formatting.
- Prompt formatting is part of the evaluation protocol, not a cosmetic detail.
- Decoding changes behavior: greedy, sampling, top-k, nucleus sampling, and temperature are not equivalent.
- Benchmark contamination is a serious issue for web-scale language models.
- Zero-shot outputs often need normalization before metric computation.
- Context length limits force truncation or retrieval-like preprocessing for long inputs.

## Limitations

- GPT-2 is a scaling and data paper as much as an architecture note.
- Zero-shot evaluation is sensitive to prompt format and benchmark leakage.
- Causal attention cannot condition on future tokens directly.
- The original report predates modern instruction tuning, RLHF, tool use, and chat interfaces.
- The paper does not provide a clean causal separation between data scale, model scale, and architecture.
- WebText-style training data makes exact benchmark exposure hard to rule out.
- Next-token prediction can generate plausible text without grounding or verification.
- A continuation interface is flexible but underspecified; users must encode task intent in text.

## Why It Matters

GPT-2 is the practical bridge from Transformer language modeling to prompt-based general-purpose LLM interfaces.

The reusable pattern is:

$$
\text{large causal LM}
+
\text{broad text corpus}
+
\text{prompt as task specification}
\rightarrow
\text{zero-shot or weakly supervised task behavior}
$$

This pattern later becomes the base for instruction-tuned LLMs, chat models, coding agents, tool-using agents, and retrieval-augmented systems.

## Connections

- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[agents/index|Agents]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/tools/tool-use|Tool use]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/gpt-3|GPT-3]]
- [[papers/architectures/index|Architecture papers]]
