---
title: Data Validation
tags:
  - systems
  - data
  - evaluation
---

# Data Validation

Data validation checks whether inputs, labels, metadata, and artifacts satisfy the assumptions of a training, evaluation, or inference workflow. It connects [[concepts/data/data-schema|data schema]], [[concepts/data/preprocessing-contract|preprocessing contract]], [[concepts/evaluation/leakage|leakage]], and [[concepts/systems/inference-contract|inference contract]].

A validation rule is a predicate:

$$
g_j(x, m, y) \in \{0, 1\}
$$

where $x$ is the input, $m$ is metadata, $y$ is the label or target when available, and $g_j=1$ means rule $j$ passes.

The dataset or batch passes if:

$$
\prod_{i=1}^{n}\prod_{j=1}^{k} g_j(x_i, m_i, y_i) = 1
$$

or, in practice, if failures are below an explicit tolerance and reviewed.

## Validation Layers

- Schema: required fields, types, shapes, units, allowed ranges, and missing-value policy.
- Identity: duplicate examples, split membership, artifact IDs, and version boundaries.
- Label semantics: endpoint, unit, threshold, censoring, weak label source, and target context.
- Distribution: class prevalence, length distribution, source mix, structure size, or request shape.
- Preprocessing: tokenizer, featurizer, graph construction, structure preparation, normalization, and fitted transforms.
- Inference: unsupported inputs, invalid outputs, timeout-prone request shapes, and logging boundary.

## Failure Modes

- Silent schema drift changes model behavior without changing code.
- Train/test leakage enters through IDs, duplicates, templates, poses, or source metadata.
- A preprocessing transform is fit on validation or test data.
- Inference accepts a request shape that was never evaluated.
- Invalid records are dropped differently across train, validation, and test.

## Checks

- Are validation rules run before training, evaluation, and batch inference?
- Are failures counted and saved as artifacts?
- Are fitted transforms learned only from training data?
- Do train, validation, test, and deployment-like inputs pass the same compatible checks?
- Is any rejected or imputed field relevant to the final claim?

## Related

- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/observability|Observability]]
