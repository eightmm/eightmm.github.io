---
title: Sequence Generation
tags:
  - tasks
  - generation
  - sequence-modeling
---

# Sequence Generation

Sequence generation produces an ordered output such as text, code, protein sequence, SMILES, action sequence, or symbolic plan.

Autoregressive generation factorizes the output distribution as:

$$
p(y_{1:T}\mid x)=\prod_{t=1}^{T}p(y_t\mid y_{<t},x)
$$

At inference time, the model chooses tokens using a decoding rule:

$$
\hat{y}_t \sim \operatorname{Decode}(p_\theta(\cdot \mid \hat{y}_{<t},x))
$$

## Decoding Choices

- Greedy decoding chooses the most likely token at each step.
- Beam search keeps several high-scoring partial sequences.
- Sampling uses temperature, top-$k$, or nucleus filtering.
- Constrained decoding restricts outputs to valid syntax or domain objects.

## Validity Function

Many sequence tasks have constraints:

$$
v(y)
\in
\{0,1\}
$$

where $v(y)=1$ means the generated sequence is valid. Examples include parseable code, valid JSON, valid SMILES, plausible protein sequence constraints, executable plans, or answers supported by evidence.

Constrained decoding changes the effective output space:

$$
\mathcal{Y}_{\mathrm{decode}}
=
\{y\in\mathcal{Y}: v(y)=1\}
$$

## Training vs Inference

Teacher-forced likelihood trains on gold prefixes:

$$
\mathcal{L}
=
-
\sum_{t=1}^{T}
\log
p_\theta(y_t\mid y_{<t},x)
$$

At inference, the model conditions on its own previous outputs:

$$
\hat{y}_t
\sim
p_\theta(\cdot\mid \hat{y}_{<t},x)
$$

This train-inference mismatch can cause error accumulation, invalid syntax, repetition, or drift away from the conditioning input.

## Evaluation Boundary

Sequence generation metrics should match the use case:

- likelihood or perplexity for distribution modeling;
- exact match or edit distance for canonical outputs;
- validity, uniqueness, diversity, and novelty for generated objects;
- execution success for code or action sequences;
- grounding and citation support for answers;
- downstream property or utility for protein, molecule, or design sequences.

## Checks

- Is the output space open-ended or constrained?
- What makes a generated sequence valid?
- Does decoding optimize likelihood, diversity, exactness, or utility?
- Are invalid molecules, invalid actions, or unsupported claims filtered?
- Is evaluation based on sequence match, functional property, human preference, or downstream execution?
- Is decoding part of the reported method or only an implementation detail?
- Are invalid generations counted in the denominator?
- Does the decoder use constraints that are also available at deployment time?
- Is diversity measured separately from quality and utility?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/systems/inference|Inference]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/generative-models/protein-design|Protein design]]
