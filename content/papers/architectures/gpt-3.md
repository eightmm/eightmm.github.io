---
title: Language Models are Few-Shot Learners
aliases:
  - papers/gpt-3
  - papers/language-models-are-few-shot-learners
  - papers/few-shot-language-models
tags:
  - papers
  - architectures
  - transformer
  - language-model
  - scaling
  - in-context-learning
---

# Language Models are Few-Shot Learners

> GPT-3 showed that a large decoder-only Transformer can use natural-language prompts and in-context examples as a task interface without gradient updates at evaluation time.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Language Models are Few-Shot Learners |
| Authors | Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei |
| Year | 2020 |
| Venue | NeurIPS 2020 |
| arXiv | [2005.14165](https://arxiv.org/abs/2005.14165) |
| Proceedings | [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |
| Status | full note started |

## One-Line Takeaway

GPT-3 keeps the decoder-only language-model objective:

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),
$$

but shows that scale changes how the interface is used:

$$
\text{task description}
+
\text{few examples}
+
\text{query}
\rightarrow
\text{next-token continuation}.
$$

The durable architecture lesson is that a causal Transformer can become a general text-conditioned task machine when the prompt itself carries the task specification.

## Question

Before GPT-3, the common transfer recipe was:

$$
\text{pretrain}
\rightarrow
\text{fine-tune on task data}
\rightarrow
\text{evaluate task-specific model}.
$$

GPT-3 asks whether a much larger autoregressive language model can instead use context as the adaptation mechanism:

$$
\text{pretrain once}
\rightarrow
\text{condition on prompt}
\rightarrow
\text{perform task without parameter update}.
$$

This is not a new attention formula. It is a major architecture-interface result:

$$
\text{causal Transformer}
+
\text{long enough prompt context}
+
\text{scale}
\Rightarrow
\text{in-context task behavior}.
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | text token prefix containing instructions, examples, and query |
| Output | next-token distribution and generated continuation |
| Backbone | dense decoder-only Transformer |
| Attention pattern | causal self-attention |
| Training objective | autoregressive next-token prediction |
| Evaluation adaptation | zero-shot, one-shot, or few-shot prompting without gradient updates |
| Main scale point | 175B-parameter GPT-3 model family endpoint |
| Main risk | benchmark contamination, prompt sensitivity, and mistaking in-context behavior for reliable reasoning |

The note belongs in Architecture Papers because it made the decoder-only Transformer a practical task interface. It should not be read as a block-level invention like attention, RoPE, or MoE routing.

## Decoder-Only Objective

The training objective is next-token prediction:

$$
\mathcal{L}_{\mathrm{LM}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
$$

The causal mask is:

$$
M_{ij}
=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
$$

Attention at a layer is:

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
$$

The architecture is therefore still in the [[concepts/architectures/decoder-only-transformer|Decoder-Only Transformer]] family.

## Prompt as Task Interface

GPT-3's important interface is:

$$
c
=
[\text{instruction};\text{examples};\text{query}].
$$

The model produces:

$$
y
\sim
p_\theta(\cdot\mid c).
$$

For a few-shot classification task, the prompt can be abstracted as:

$$
c
=
(x_1,y_1),(x_2,y_2),\ldots,(x_k,y_k),x_{\mathrm{test}}.
$$

The model then predicts the continuation:

$$
\hat{y}
=
\arg\max_y p_\theta(y\mid c).
$$

No gradient update is applied at evaluation time:

$$
\theta_{\mathrm{eval}}=\theta_{\mathrm{pretrain}}.
$$

That is the key contrast with fine-tuning.

## Zero-Shot, One-Shot, Few-Shot

| Setting | Prompt Contents | Parameter Update? | What It Tests |
| --- | --- | --- | --- |
| zero-shot | instruction or task text only | no | task inference from language |
| one-shot | one demonstration plus query | no | format induction from one example |
| few-shot | several demonstrations plus query | no | in-context pattern use |
| fine-tuning | labeled task dataset | yes | task-specific parameter adaptation |

The paper's core claim is about prompt-conditioned evaluation, not supervised fine-tuning.

## Scaling View

GPT-3 is part of a model family. The central empirical pattern is:

$$
\text{larger model}
\Rightarrow
\text{stronger few-shot behavior}.
$$

This should be read with three variables separated:

$$
\text{architecture}
\quad
\text{data}
\quad
\text{compute}.
$$

The architecture family is decoder-only Transformer. The paper's claim is that scaling this family improves task-agnostic performance under prompt conditioning.

## Relation to GPT-2 and LLaMA

| Paper | Main Role | Interface |
| --- | --- | --- |
| [GPT-2](/papers/architectures/gpt-2) | zero-shot decoder-only LM transfer | task as text continuation |
| GPT-3 | few-shot/in-context scaling milestone | task as prompt with demonstrations |
| [LLaMA](/papers/architectures/llama) | efficient open-weight foundation model recipe | modern decoder-only pretraining recipe |

The rough progression is:

$$
\text{GPT-2}
\rightarrow
\text{GPT-3}
\rightarrow
\text{modern open decoder-only LLM recipes}.
$$

GPT-3 sits between GPT-2's prompt-as-continuation framing and later foundation-model recipes that standardize RMSNorm, RoPE, SwiGLU, larger data mixtures, instruction tuning, and tool use.

## What Is Architectural Here?

GPT-3 is easy to misfile as only a scaling or capability paper. For this wiki, keep it in architecture because it changes the practical contract of the decoder-only Transformer:

$$
\text{model input}
\neq
\text{only raw text prefix};
$$

it becomes:

$$
\text{task specification}
+
\text{demonstrations}
+
\text{query}.
$$

That means the architecture is evaluated not just as a sequence model, but as a conditional computation system whose behavior is controlled through the context window.

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| larger autoregressive LMs improve few-shot performance | model-size sweep across benchmarks | scale improves prompt-conditioned task behavior | data, task choice, and prompt formatting matter |
| no evaluation-time gradient update is needed for many tasks | zero/one/few-shot protocol | prompt context can act as adaptation signal | not equivalent to robust task learning |
| performance is broad across NLP tasks | multi-benchmark evaluation | decoder-only LM can serve as a general interface | some tasks remain weak or brittle |
| generated text becomes hard to distinguish in some settings | human evaluation of samples | fluent generation reaches a notable threshold | social and measurement risks are central |

## Implementation Reading

Check:

- context window length and whether demonstrations fit;
- prompt format and demonstration ordering;
- whether evaluation uses exact-match, likelihood ranking, or free generation;
- whether labels are verbalized consistently;
- whether train/test contamination was audited;
- whether results compare zero-shot, one-shot, few-shot, and fine-tuned baselines separately;
- whether the claim is about architecture, scale, data, or prompting.

For reproduction, the hard part is not writing a decoder-only Transformer equation. It is reproducing scale, data mixture, evaluation protocol, and contamination checks.

## Limitations

- The architecture family is not new; the contribution is the scaled interface behavior.
- Prompt sensitivity can dominate conclusions on individual tasks.
- In-context examples do not guarantee robust reasoning or stable calibration.
- Benchmark contamination and web-scale data overlap are hard to rule out completely.
- Few-shot performance is not a substitute for controlled domain evaluation in high-stakes settings.

## Common Misreadings

| Misreading | Correction |
| --- | --- |
| "GPT-3 invented the Transformer." | It scales a decoder-only Transformer and changes the task interface through prompting. |
| "Few-shot means parameter-efficient fine-tuning." | In GPT-3, few-shot examples are in the context; parameters are not updated. |
| "Prompting proves reasoning." | It shows prompt-conditioned behavior; reasoning claims need stricter tests. |
| "Architecture explains all gains." | Data, compute, scale, prompt format, and benchmark protocol are inseparable. |

## What to Remember

GPT-3 is the canonical architecture-interface paper for decoder-only LLMs:

$$
\text{causal LM}
+
\text{scale}
+
\text{in-context examples}
\rightarrow
\text{few-shot task behavior}.
$$

In this wiki, keep it as the bridge between [[papers/architectures/gpt-2|GPT-2]], modern decoder-only architecture recipes like [[papers/architectures/llama|LLaMA]], and agent/tool-use systems that treat language-model context as a working interface.

## Links

- [[concepts/architectures/decoder-only-transformer|Decoder-Only Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/llama|LLaMA]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[agents/index|Agents]]
