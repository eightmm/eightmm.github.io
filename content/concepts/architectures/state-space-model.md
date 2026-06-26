---
title: State-Space Models
tags:
  - architectures
  - state-space-model
  - sequence-modeling
---

# State-Space Models

State-space models are sequence architectures that model long-range dependencies through recurrent state updates or structured sequence transformations.

A simple continuous state-space form is:

$$
\frac{dh(t)}{dt} = Ah(t) + Bx(t),
\qquad
y(t) = Ch(t)
$$

Sequence models use discretized or structured versions of this update to mix information over long contexts.

A discrete-time form is:

$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t,
\qquad
y_t = Ch_t + Dx_t
$$

The matrices may be constrained or parameterized so the sequence can be scanned efficiently.

## Key Ideas

- A state summarizes past sequence information and is updated as new inputs arrive.
- Structured parameterization can make long sequence mixing efficient compared with full [[concepts/architectures/attention|attention]].
- Some implementations behave like recurrent scans; others use parallel sequence transforms during training.
- State size, discretization, gating, and input-dependent parameters define what information can be retained.
- SSMs are often compared with [[concepts/architectures/rnn|RNNs]] and [[concepts/architectures/transformer|Transformers]] because all three solve sequence mixing differently.
- [[concepts/architectures/mamba|Mamba]] is best treated as a selective SSM rather than a separate top-level architecture family.

## Practical Checks

- Check whether the model is causal, bidirectional, or wrapped with extra context mixing.
- Track the effective context length claimed by the implementation, not just the theoretical state.
- Inspect how masks, padding, and variable-length sequences are handled.
- For protein tasks, verify whether residue order alone is modeled or whether structure, contacts, or family splits are also used.

## Related

- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/rnn|RNN]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[entities/protein|Protein]]
