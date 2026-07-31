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
| Status | full paper note |

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

## Pre-training Input Contract

BERT does not receive an arbitrary text string directly. It receives a packed sequence with explicit segment structure:

$$
[\text{CLS}],\; A,\; [\text{SEP}],\; B,\; [\text{SEP}].
$$

For a single sentence, segment $B$ is empty. The embedding at position $i$ combines three lookup tables:

$$
e_i
=
e_i^{\text{token}}
+ e_i^{\text{segment}}
+ e_i^{\text{position}}.
$$

The segment embedding is a small but important interface decision. It tells the same shared encoder whether a token belongs to sentence $A$ or sentence $B$ without requiring a separate encoder for each side. The `[SEP]` tokens provide explicit boundaries to the self-attention stack.

The representation contract is therefore:

| Input form | Main state used | Typical output |
| --- | --- | --- |
| one segment | token states and `[CLS]` | classification or tagging |
| two segments | shared contextual states with segment IDs | entailment, matching, pair classification |
| question plus context | per-token contextual states | start/end span prediction |

This packing convention is part of the architecture. Replacing it with an unmarked concatenation changes the information available to the encoder and the downstream head.

## Masked Language Modeling in Detail

Let $S$ be the set of selected token positions. In the original recipe, approximately 15 percent of token positions are selected for prediction. For each selected position, the input corruption is sampled from three cases:

| Case | Approximate fraction of selected positions | Input shown to encoder |
| --- | ---: | --- |
| mask | 80 percent | `[MASK]` token |
| unchanged | 10 percent | original token |
| random replacement | 10 percent | random vocabulary token |

The target remains the original token in all three cases. The objective is:

$$
\mathcal{L}_{\text{MLM}}
=
-\frac{1}{|S|}
\sum_{i\in S}
\log p_\theta(x_i\mid x_{\setminus S}).
$$

The unchanged and random-replacement cases reduce the pre-training/fine-tuning mismatch caused by a special `[MASK]` token that is absent from ordinary downstream inputs. They also mean that the model cannot assume that every selected target position is visibly marked as corrupted.

MLM is not ordinary left-to-right language modeling. The attention graph for an unmasked token can include both left and right context, while the loss is applied only to selected positions. The useful separation is:

$$
\text{bidirectional context}
\ne
\text{bidirectional generation}.
$$

BERT learns contextual representations using both directions; it does not define a left-to-right procedure for generating an arbitrary paragraph.

## Next Sentence Prediction

For the original next-sentence objective, a pair of segments is labeled as either a true consecutive pair or a randomly sampled pair. The binary loss is:

$$
\mathcal{L}_{\text{NSP}}
=
-y\log \hat{y}
-(1-y)\log(1-\hat{y}).
$$

The total objective is commonly written as:

$$
\mathcal{L}_{\text{pretrain}}
=
\mathcal{L}_{\text{MLM}}
+
\mathcal{L}_{\text{NSP}}.
$$

NSP should be read as a historical task-interface choice, not as a necessary property of every encoder. Later work changed or removed it, used sentence-order alternatives, or trained with larger contiguous spans while retaining the encoder-only backbone. This makes BERT useful as a decomposition point:

$$
\text{encoder architecture}
\quad\text{versus}\quad
\text{pre-training objective recipe}.
$$

When comparing BERT with later encoders, keep those two axes separate.

## Representation Flow

For input length $T$ and hidden width $d$, the encoder produces:

$$
H^{(L)}
=
f_{\theta}^{(L)}\circ\cdots\circ f_{\theta}^{(1)}(E),
\qquad
H^{(L)}\in\mathbb{R}^{T\times d}.
$$

Each layer mixes information across the entire sequence through self-attention and then applies a token-wise feed-forward transformation. The contextual state at position $i$ is not a fixed feature extractor output:

$$
h_i^{(L)}
=
f_\theta(x_1,\ldots,x_T)_i.
$$

The same token can receive a different representation when the surrounding sentence changes. This context dependence is the reason BERT can support token labeling, span extraction, and sentence-level classification from one backbone.

For masked prediction, a small output transformation maps the selected hidden states to vocabulary logits:

$$
\ell_i=W_{\text{MLM}}h_i+b_{\text{MLM}},
\qquad
p(x_i\mid x_{\setminus S})=\operatorname{softmax}(\ell_i).
$$

For a downstream classification task, the same encoder state is connected to a task-specific head instead:

$$
\hat{y}
=
\operatorname{softmax}(W_{\text{task}}h_{\text{[CLS]}}+b_{\text{task}}).
$$

The backbone and head therefore have different lifetimes: the pre-training head is discarded or replaced, while the encoder is transferred.

## Model Scale as an Architecture Variable

The original paper presents base and large configurations. Their important distinction is not the model name but the scaling axes:

| Axis | Base-style configuration | Large-style configuration | Architectural effect |
| --- | --- | --- | --- |
| encoder depth | fewer blocks | more blocks | more sequential representation transformations |
| hidden width | narrower states | wider states | larger token representation and projection matrices |
| attention heads | fewer heads | more heads | more parallel relation subspaces |
| parameters | lower | higher | more capacity and higher pre-training cost |

For a Transformer layer, the dominant dense projections scale approximately with width and the attention interaction scales with sequence length:

$$
C_{\text{layer}}
\approx
O(T^2d)+O(Td^2).
$$

This explains two boundaries of the original BERT design:

1. increasing sequence length is expensive because of the $T^2$ attention term;
2. increasing hidden width is expensive because of the $Td^2$ projection and feed-forward terms.

Model size alone is not a complete explanation for transfer quality. Corpus size, tokenization, optimization schedule, sequence-length curriculum, and fine-tuning hyperparameters are part of the effective system.

## Architecture Versus Objective

The BERT contribution is often summarized as “MLM plus Transformer,” but the reusable architecture can be decomposed more precisely:

| Layer of the system | BERT choice | Can later work change it? |
| --- | --- | --- |
| token interface | WordPiece-style subword tokens and special separators | yes, with other tokenizers or byte-level units |
| backbone | stacked bidirectional Transformer encoder | yes, with different attention or state operators |
| corruption | selected-token masking | yes, with spans, replaced tokens, or denoising schemes |
| auxiliary target | next-sentence classification | yes, remove or replace it |
| transfer interface | `[CLS]`, token states, span heads | yes, pool, retrieve, or attach task-specific modules |

This decomposition is useful when reading protein, image, or multimodal encoders. A masked objective can be transferred while changing the token unit and backbone; conversely, the encoder can be retained while replacing MLM with another representation objective.

## Evidence and Claim Boundaries

The headline benchmark numbers in the paper support the claim that BERT was a strong transfer recipe on the evaluated language-understanding tasks. They do not independently prove each of the following stronger claims:

| Strong statement | What would be needed to support it |
| --- | --- |
| bidirectionality alone caused the gain | controlled objective, data, and parameter comparisons |
| NSP is necessary | ablations across datasets and later training recipes |
| `[CLS]` is a universal sentence embedding | retrieval and semantic similarity evaluations, not only classification |
| pre-training always improves transfer | task- and domain-matched baselines with equal compute |
| BERT is suitable for generation | an explicit generation architecture and decoding evaluation |

The paper's most durable evidence is the combination of a shared encoder, simple adaptation heads, and gains across several task families. The exact objective recipe should remain historically important without being treated as immutable.

## Ablations and Reading Questions

When reading or reproducing BERT, ask:

- Does the comparison use the same tokenizer and vocabulary coverage?
- Are the number of pre-training updates, tokens, and sequence lengths matched?
- Is the baseline bidirectional, left-to-right, or shallowly contextual?
- Are MLM and NSP changed together, or is one isolated?
- Is a result from frozen probing, full fine-tuning, or an intermediate setting?
- Does the task use `[CLS]`, per-token states, or span logits?
- Are sentence pairs sampled in a way that makes NSP artificially easy?
- Does the downstream task exceed the 512-token sequence limit?

Useful ablation axes include:

| Ablation | Isolates |
| --- | --- |
| remove NSP | value of the auxiliary sentence-pair signal |
| alter mask replacement proportions | sensitivity to the pre-training/fine-tuning mismatch |
| replace token masking with span masking | dependence on local versus contiguous corruption |
| reduce pre-training data or steps | data/compute scaling versus architecture |
| freeze encoder and train a probe | quality of the representation without task adaptation |
| compare encoder-only and decoder-only attention masks | contextual representation versus autoregressive prediction |

## Failure Modes in Practice

1. **Tokenizer mismatch**: a downstream domain uses many rare or fragmented subwords, so the model spends capacity on tokenization artifacts.
2. **Mask leakage**: preprocessing exposes target information through duplicated text, labels, or unmasked copies.
3. **Sentence-pair shortcut**: NSP examples contain topic or document-boundary artifacts that make the auxiliary task easier than discourse understanding.
4. **Pooler overinterpretation**: the `[CLS]` state is treated as a universal embedding without a matching objective or evaluation.
5. **Context truncation**: important evidence falls outside the maximum input window, silently changing the task.
6. **Fine-tuning variance**: small data or unstable hyperparameters make benchmark gains sensitive to seed and schedule.
7. **Generation mismatch**: an encoder checkpoint is used as if it were a causal decoder, even though its attention and training objective do not define that interface.

For a minimal implementation test, verify that changing the right-hand context can change the representation of a token in the left-hand segment. Then verify that a causal attention mask produces a different model contract rather than merely a different tensor shape.

## Relation to Later Encoder Families

BERT is an anchor, not the final encoder design:

| Later family | What it changes relative to BERT |
| --- | --- |
| RoBERTa-style training | data, batch, schedule, and objective recipe while keeping an encoder backbone |
| ALBERT-style factorization | parameter sharing and embedding factorization for efficiency |
| DeBERTa | disentangled content/position representations and attention design |
| Long-context encoders | attention pattern or memory interface to reduce the quadratic window limit |
| protein language encoders | token unit and corpus, while reusing masked representation learning ideas |
| vision masked encoders | patch units and reconstruction target, often with an asymmetric decoder |

The comparison with [[papers/architectures/masked-autoencoders-are-scalable-vision-learners|MAE]] is especially instructive. Both hide part of the input, but BERT keeps the full sequence in the encoder and predicts discrete token identities, while MAE removes masked patches from the expensive encoder and reconstructs continuous patch targets with a lightweight decoder.

## Reproduction Checklist

- [ ] define tokenizer, vocabulary, special tokens, and maximum sequence length;
- [ ] verify segment IDs and `[CLS]`/`[SEP]` placement;
- [ ] implement the selected-token mask with the recorded replacement proportions;
- [ ] ensure MLM loss is computed only at selected positions;
- [ ] record whether NSP is used, removed, or replaced;
- [ ] record depth, width, head count, and normalization placement;
- [ ] separate pre-training output heads from downstream task heads;
- [ ] distinguish frozen probing from full fine-tuning;
- [ ] report token count, update count, batch size, and compute budget;
- [ ] test long-input truncation and sentence-pair formatting;
- [ ] compare against a causal and a non-pretrained baseline where relevant.

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
