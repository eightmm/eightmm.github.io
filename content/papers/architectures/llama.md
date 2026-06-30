---
title: LLaMA
aliases:
  - papers/llama
  - papers/llama-open-and-efficient-foundation-language-models
tags:
  - papers
  - architectures
  - transformer
  - language-model
  - foundation-model
---

# LLaMA

> The paper made the modern efficient open-weight decoder-only language model recipe concrete: train smaller dense Transformers longer, on public data, with a stable pre-normalized architecture.

## Metadata

| Field | Value |
| --- | --- |
| Paper | LLaMA: Open and Efficient Foundation Language Models |
| Authors | Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample |
| Year | 2023 |
| Venue | arXiv preprint |
| arXiv | [2302.13971](https://arxiv.org/abs/2302.13971) |
| Meta page | [Meta AI publication](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/) |
| Status | full note started |

## One-Line Takeaway

LLaMA is important less because it invented a single new block and more because it established a practical decoder-only foundation model recipe: public-data pretraining, compute-conscious model sizes, pre-normalization, RMSNorm, SwiGLU feed-forward layers, rotary position embeddings, and careful evaluation across many tasks.

## Question

Large language model progress before LLaMA was dominated by very large closed models. The architectural and systems question was:

> How strong can a dense decoder-only Transformer become if trained on more tokens, with an efficient architecture recipe, without relying on private training data?

The practical question:

$$
\text{smaller model}
+
\text{more training tokens}
+
\text{stable Transformer recipe}
\Rightarrow
\text{strong foundation model}
$$

The paper should be read as an architecture-and-training recipe paper, not only as a model release.

## Main Claim

The narrowed claim:

$$
\text{decoder-only Transformer}
+
\text{public-data trillion-token training}
+
\text{compute-efficient scaling}
\Rightarrow
\text{competitive foundation models}
$$

The paper reports a family of dense language models from 7B to 65B parameters. The most important architectural message is not that the Transformer changed completely. It is that a specific set of now-standard decoder-only choices can support strong language models under a public-data constraint.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | token sequence |
| Output | next-token probability distribution |
| Backbone | decoder-only Transformer |
| Attention pattern | causal self-attention |
| Normalization | pre-normalization with RMSNorm |
| Position handling | rotary positional embeddings |
| Feed-forward block | SwiGLU-style gated MLP |
| Training objective | autoregressive next-token prediction |
| Main scaling route | more tokens for smaller models, public data mixture |

The core interface is the same as [[papers/architectures/gpt-2|GPT-2]]:

$$
p_\theta(x)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t}).
$$

The training loss is:

$$
\mathcal{L}_{\text{LM}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t}).
$$

The architectural difference is in the modernized block choices and the scaling recipe.

## Decoder-Only Transformer

LLaMA uses a causal Transformer stack. For a sequence of token embeddings:

$$
X_0 \in \mathbb{R}^{T \times d},
$$

each layer maps:

$$
X_{\ell}
\rightarrow
X_{\ell+1}.
$$

A simplified pre-normalized decoder block is:

$$
U_{\ell}
=
X_{\ell}
+
\operatorname{Attn}_{\ell}(\operatorname{Norm}(X_{\ell})),
$$

$$
X_{\ell+1}
=
U_{\ell}
+
\operatorname{FFN}_{\ell}(\operatorname{Norm}(U_{\ell})).
$$

This is different from a post-norm form:

$$
X_{\ell+1}
=
\operatorname{Norm}(X_{\ell} + F(X_{\ell})).
$$

Pre-normalization improves optimization stability in deep Transformer stacks because each sublayer sees normalized inputs before the residual addition.

## Causal Self-Attention

For hidden states:

$$
H \in \mathbb{R}^{T \times d},
$$

attention forms:

$$
Q = HW_Q,\quad K = HW_K,\quad V = HW_V.
$$

For a single head:

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
+ M
\right)V,
$$

where:

- $d_h$ is the head dimension;
- $M_{ij}=0$ if token $i$ may attend to token $j$;
- $M_{ij}=-\infty$ if $j>i$ under causal masking.

The mask enforces:

$$
p(x_t \mid x_{\leq T})
\rightarrow
p(x_t \mid x_{<t}).
$$

This is the same basic next-token contract as GPT-style models, but LLaMA's block choices became a reference point for later open-weight models.

## RMSNorm

[[papers/architectures/root-mean-square-layer-normalization|RMSNorm]] is one of the small but important block choices in the LLaMA recipe. LayerNorm normalizes by subtracting the mean and dividing by the standard deviation:

$$
\operatorname{LayerNorm}(x)
=
\gamma
\odot
\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}
+
\beta.
$$

RMSNorm removes mean-centering and normalizes by root mean square:

$$
\operatorname{RMSNorm}(x)
=
g
\odot
\frac{x}{\operatorname{RMS}(x)+\epsilon},
$$

where:

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}
x_i^2
}.
$$

Why this matters architecturally:

| Choice | Effect |
| --- | --- |
| no mean subtraction | simpler normalization path |
| scale normalization | controls activation magnitude |
| pre-norm placement | stabilizes deep residual stacks |
| fewer operations than full LayerNorm | useful for large-scale training and inference |

The point is not that RMSNorm alone explains LLaMA performance. It is one part of a stable decoder-only block recipe.

## Rotary Positional Embeddings

Transformers need some way to encode token position. Absolute position embeddings add a learned or fixed vector:

$$
h_t
\leftarrow
h_t + p_t.
$$

Rotary positional embeddings instead rotate query and key vectors as a function of position before attention:

$$
q_t' = R_t q_t,
\quad
k_s' = R_s k_s,
$$

where $R_t$ is a block-diagonal rotation matrix parameterized by token position $t$.

The attention score becomes:

$$
(q_t')^\top k_s'
=
(R_t q_t)^\top (R_s k_s)
=
q_t^\top R_t^\top R_s k_s.
$$

Because:

$$
R_t^\top R_s
=
R_{s-t},
$$

the dot product can express relative displacement between positions.

Reading contract:

| Positional method | What is injected |
| --- | --- |
| learned absolute embedding | a position vector added to hidden state |
| sinusoidal absolute embedding | fixed position basis added to hidden state |
| rotary embedding | position-dependent rotation applied to queries and keys |

For LLaMA-style models, RoPE became a default because it fits causal attention naturally and preserves the dot-product attention interface.

## SwiGLU Feed-Forward Network

[[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]] is the paper-level reference for the gated FFN family behind SwiGLU. The standard Transformer feed-forward block is:

$$
\operatorname{FFN}(x)
=
W_2 \phi(W_1 x),
$$

where $\phi$ is often ReLU or GELU.

A gated feed-forward block uses one projection to produce content and another to produce a gate:

$$
\operatorname{GLU}(x)
=
(xW_a)
\odot
\sigma(xW_b).
$$

SwiGLU uses the Swish activation:

$$
\operatorname{Swish}(z)
=
z\sigma(z),
$$

and a common form is:

$$
\operatorname{SwiGLU}(x)
=
(xW_a)
\odot
\operatorname{Swish}(xW_b).
$$

The output projection maps back to model dimension:

$$
\operatorname{FFN}_{\text{SwiGLU}}(x)
=
W_o
\left[
(xW_a)
\odot
\operatorname{Swish}(xW_b)
\right].
$$

Architectural reading:

| FFN style | Inductive effect |
| --- | --- |
| ReLU/GELU MLP | feature expansion and nonlinear mixing |
| GLU family | feature expansion with multiplicative gating |
| SwiGLU | smoother gate used in many modern LLM blocks |

The gated MLP is a token-wise computation. It does not mix sequence positions; attention does that. It changes how each token representation transforms after contextual mixing.

## Tokenization and Vocabulary Interface

LLaMA uses subword tokenization. The exact tokenizer is not the architecture contribution, but tokenization defines the model's input/output alphabet:

$$
\text{text}
\rightarrow
(x_1,\ldots,x_T),
\quad
x_t \in \{1,\ldots,V\}.
$$

The model outputs logits:

$$
z_t \in \mathbb{R}^{V}.
$$

Next-token probabilities are:

$$
p_\theta(x_{t+1}=v \mid x_{\leq t})
=
\frac{\exp z_{t,v}}
{\sum_{u=1}^{V}\exp z_{t,u}}.
$$

Tokenization matters because:

- vocabulary size affects embedding and output projection size;
- token boundaries change sequence length;
- long words, code, and multilingual text can become different tokenization regimes;
- token-level loss is not always comparable across tokenizers.

## Public Data Constraint

The LLaMA paper emphasizes training on publicly available datasets. For architecture reading, this matters because it changes what the result means.

The claim is not:

$$
\text{secret dataset}
\Rightarrow
\text{strong model}.
$$

The claim is closer to:

$$
\text{known architecture recipe}
+
\text{public data mixture}
+
\text{large token budget}
\Rightarrow
\text{strong open research model}.
$$

This made the paper useful as a reference point for researchers who could not inspect closed-model training data.

## Scaling Reading

LLaMA follows the post-Chinchilla intuition that many large models were under-trained relative to parameter count. The practical architecture lesson:

$$
\text{more parameters only}
\neq
\text{best compute allocation}.
$$

A simplified compute view:

$$
C
\propto
N_{\text{params}}
\cdot
N_{\text{tokens}},
$$

where:

- $C$ is approximate training compute;
- $N_{\text{params}}$ is parameter count;
- $N_{\text{tokens}}$ is number of training tokens.

For a fixed compute budget, one can trade:

$$
\text{larger model, fewer tokens}
\quad
\text{vs.}
\quad
\text{smaller model, more tokens}.
$$

The paper helped popularize the second direction for open foundation models.

## Model Family

The paper presents several sizes rather than a single architecture instance.

| Family Aspect | Reading |
| --- | --- |
| 7B | deployable research-scale dense LM |
| 13B | stronger small model; important comparison point against much larger models |
| 33B | mid-scale foundation model |
| 65B | large open research model of the original family |

The same architectural recipe across sizes makes the paper useful for comparing scaling behavior without changing model family.

## What Changed Relative to GPT-2

| Axis | GPT-2 | LLaMA |
| --- | --- | --- |
| backbone | decoder-only Transformer | decoder-only Transformer |
| objective | next-token prediction | next-token prediction |
| position | learned absolute positions | rotary positional embeddings |
| normalization | LayerNorm-style Transformer recipe | pre-norm RMSNorm recipe |
| FFN | standard Transformer MLP | SwiGLU gated MLP |
| central claim | zero-shot behavior from web-scale LM | efficient open foundation models from public data |
| model release role | early GPT-style public reference | central open-weight research reference |

This comparison is important. LLaMA did not replace the decoder-only Transformer paradigm. It made a newer implementation recipe the practical baseline for open LLM work.

## What Changed Relative to BERT and T5

| Model | Architecture Contract | Main Use |
| --- | --- | --- |
| [[papers/architectures/bert|BERT]] | encoder-only masked language model | representation learning and fine-tuning |
| [[papers/architectures/t5|T5]] | encoder-decoder text-to-text model | supervised and multitask seq2seq transfer |
| LLaMA | decoder-only causal language model | generative continuation, prompting, instruction tuning base |

LLaMA is not a general replacement for encoder-only or encoder-decoder architectures. It is the canonical path for open autoregressive foundation models.

## Evidence Reading

The paper evaluates language understanding, reasoning, commonsense, reading comprehension, and mathematical benchmarks. For a paper note, the exact benchmark scores are less important than what the evidence is trying to support:

| Evidence Type | What It Supports | What It Does Not Prove |
| --- | --- | --- |
| comparison to larger models | smaller public-data models can be competitive | architecture alone caused the gain |
| many benchmark categories | broad transfer behavior | robust real-world reliability |
| model family scaling | same recipe scales across sizes | optimality of every hyperparameter |
| public-data training | strong models need not depend entirely on private corpora | all data provenance issues are solved |

The key reading discipline:

$$
\text{benchmark score}
\neq
\text{isolated architecture proof}.
$$

The benchmark result is an outcome of architecture, data, compute, tokenizer, training recipe, and evaluation protocol.

## Implementation Notes

When implementing a LLaMA-style block, separate the concerns:

| Component | Implementation Question |
| --- | --- |
| RMSNorm | is normalization before attention and before FFN? |
| RoPE | are rotations applied to query and key, not value? |
| causal mask | does token $t$ avoid attending to future tokens? |
| SwiGLU | are hidden dimensions adjusted to keep parameter count comparable? |
| KV cache | does inference reuse past keys and values? |
| tokenizer | are input IDs compatible with the checkpoint vocabulary? |

Autoregressive inference without caching recomputes attention keys and values for previous tokens. With KV cache:

$$
K_{\leq t}
=
[K_{\leq t-1};K_t],
\quad
V_{\leq t}
=
[V_{\leq t-1};V_t].
$$

The next step attends over cached history:

$$
\operatorname{Attn}(q_t,K_{\leq t},V_{\leq t}).
$$

This is an inference-systems concern, but it is inseparable from decoder-only architecture in deployment.

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| treating LLaMA as a new Transformer type | misses that it is mainly a modern decoder-only recipe |
| attributing gains only to RMSNorm/RoPE/SwiGLU | ignores data and compute allocation |
| comparing token-level loss across tokenizers naively | tokenization changes the unit of prediction |
| ignoring public-data mixture | weakens interpretation of what the paper demonstrates |
| assuming benchmark breadth means reliability | benchmark coverage is not the same as robust behavior |
| mixing base model and instruction-tuned model claims | architecture paper concerns the base pretraining recipe |

## Common Misreadings

### "LLaMA invented decoder-only Transformers."

No. GPT-style decoder-only Transformers came earlier. LLaMA popularized a strong open foundation-model recipe.

### "The architecture explains the whole performance jump."

No. Performance depends on architecture, data, token budget, optimization, and evaluation choices.

### "Public data means the model is fully transparent."

No. Public-data training is more inspectable than private-data training, but exact preprocessing, filtering, deduplication, and data governance still matter.

### "A smaller model beating a larger model means parameters do not matter."

No. It means parameter count alone is a bad comparison axis when token budget and training recipe differ.

## Why It Matters

LLaMA became a reference architecture because it made several choices converge:

- decoder-only causal language modeling remained the dominant generative interface;
- pre-normalized RMSNorm blocks became common in open LLMs;
- RoPE became a standard positional method for causal attention;
- gated MLPs became a standard FFN choice;
- public-data foundation models became credible research infrastructure;
- open-weight models became a practical base for fine-tuning, alignment, retrieval, agents, and domain adaptation.

For this wiki, LLaMA is the bridge between:

$$
\text{GPT-2 as decoder-only LM proof}
\rightarrow
\text{modern open foundation model recipe}.
$$

## Later-Paper Checklist

When reading later LLaMA-family or open LLM papers, check:

- Is the model still dense decoder-only, or does it use MoE?
- Does it keep RMSNorm, RoPE, and gated FFN?
- Does it change context length?
- Does it use grouped-query or multi-query attention?
- What data mixture changed?
- How much of the gain comes from tokenizer changes?
- Are results for base models, instruction-tuned models, or RLHF/DPO models?
- Are comparisons compute-matched or only parameter-matched?
- Are benchmarks contaminated by training data?

## Limitations

The paper is not a clean architecture ablation. It does not isolate each block choice under a fully controlled compute/data setup. It is better read as a strong model-family recipe and evidence that open, public-data foundation models can be competitive.

The note should therefore avoid claims like:

$$
\text{RMSNorm}
\Rightarrow
\text{LLaMA performance}.
$$

The more defensible claim is:

$$
\text{modern decoder-only recipe}
+
\text{large public-data pretraining}
+
\text{compute-conscious scaling}
\Rightarrow
\text{strong open foundation model family}.
$$

## Connections

- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/t5|T5]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/switch-transformer|Switch Transformer]]
- [[papers/architectures/index|Architecture papers]]
