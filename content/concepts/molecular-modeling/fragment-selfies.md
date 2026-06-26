---
title: Fragment-SELFIES
tags:
  - concept
  - molecular-modeling
  - representation
status: stub
---

# Fragment-SELFIES

## Definition

Fragment-SELFIES is a fragment-level molecular string representation in the SELFIES family. Instead of emitting only atom-level tokens, a model emits chemically meaningful fragments and attachment structure, which can shorten sequences and make generation closer to medicinal-chemistry building blocks.

## Why It Matters

In [[concepts/generative-models/molecular-generation|molecular generation]], standard SMILES can produce invalid strings, and atom-level SELFIES guarantees validity but generates long sequences for large molecules. Fragment-SELFIES offers a middle ground: shorter sequences with chemically interpretable tokens and maintained validity guarantees.

## Modeling Contract

Let a molecule $M$ be decomposed into fragments:

$$
M
\rightarrow
(f_1,\ldots,f_n,A)
$$

where $f_i$ are fragment tokens and $A$ records attachment structure. A valid decoder must map the token sequence back to a chemically valid molecular graph:

$$
\operatorname{decode}(f_1,\ldots,f_n,A)
\in
\mathcal{G}_{\mathrm{valid}}
$$

The hard part is not only syntactic validity, but whether the fragment vocabulary covers the target chemistry without hiding stereochemistry, charge, tautomer, or rare functional groups.

## Checks

- Is the fragment vocabulary fixed, learned, or dataset-specific?
- Does tokenization preserve stereochemistry, charge, isotopes, and aromaticity?
- Are attachment points unambiguous?
- Does validity guarantee imply chemically useful molecules or only graph-valid molecules?
- Is novelty measured against fragment recombination as well as full-molecule identity?

## Related Papers

- [[papers/generative-models/molexar|Molexar]]

## Related Concepts

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]

## Open Questions

- How does generation quality compare to atom-level SELFIES for property-conditioned tasks?
- Is there a standard fragment vocabulary, or is it task-dependent?
- Does fragment-level tokenization lose fine-grained control over individual atom placement?
