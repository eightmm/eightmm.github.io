---
title: State-Space Models
tags:
  - architectures
  - state-space-model
  - sequence-modeling
---

# State-Space Models

State-space models are sequence architectures that model long-range dependencies through recurrent state updates or structured sequence transformations.

## Key Ideas

- A state summarizes past sequence information and is updated as new inputs arrive.
- Structured parameterization can make long sequence mixing efficient compared with full [[concepts/architectures/attention|attention]].
- Some implementations behave like recurrent scans; others use parallel sequence transforms during training.
- State size, discretization, gating, and input-dependent parameters define what information can be retained.
- SSMs are often compared with [[concepts/architectures/rnn|RNNs]] and [[concepts/architectures/transformer|Transformers]] because all three solve sequence mixing differently.

## Practical Checks

- Check whether the model is causal, bidirectional, or wrapped with extra context mixing.
- Track the effective context length claimed by the implementation, not just the theoretical state.
- Inspect how masks, padding, and variable-length sequences are handled.
- For protein tasks, verify whether residue order alone is modeled or whether structure, contacts, or family splits are also used.

## Related

- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/rnn|RNN]]
- [[research/protein-modeling/index|Protein modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[entities/protein|Protein]]
