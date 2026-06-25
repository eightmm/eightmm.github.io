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

## Checks

- Is the output space open-ended or constrained?
- What makes a generated sequence valid?
- Does decoding optimize likelihood, diversity, exactness, or utility?
- Are invalid molecules, invalid actions, or unsupported claims filtered?
- Is evaluation based on sequence match, functional property, human preference, or downstream execution?
- Is decoding part of the reported method or only an implementation detail?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/systems/inference|Inference]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/generative-models/protein-design|Protein design]]
