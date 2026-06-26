---
title: Fragment-SELFIES
tags:
  - concept
  - molecular-modeling
  - representation
---

# Fragment-SELFIES

## Definition

Fragment-SELFIES is a fragment-level molecular string representation in the SELFIES family. Instead of emitting only atom-level tokens, a model emits chemically meaningful fragments and attachment structure, which can shorten sequences and make generation closer to medicinal-chemistry building blocks.

## Why It Matters

In [[concepts/generative-models/molecular-generation|molecular generation]], standard SMILES can produce invalid strings, and atom-level SELFIES guarantees validity but generates long sequences for large molecules. Fragment-SELFIES offers a middle ground: shorter sequences with chemically interpretable tokens and maintained validity guarantees.

It is useful to treat Fragment-SELFIES as a representation choice, not as a complete model. A paper using it still needs a clear task, objective, data split, novelty rule, and chemical-state policy.

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

## Representation Boundary

Fragment tokenization changes what the model can easily learn:

| Boundary | Meaning | Risk |
| --- | --- | --- |
| Fragment vocabulary | allowed building blocks | rare chemistry may be unreachable |
| Attachment grammar | how fragments connect | ambiguous attachment can create invalid or unintended graphs |
| Chemical state | stereo, charge, aromaticity, tautomer, protonation | validity can hide state loss |
| Sequence policy | canonical, randomized, or augmented encoding | evaluation may reward tokenization artifacts |
| Reconstruction rule | how tokens become a molecular graph | decoder validity is not the same as usefulness |

For a generative model, the effective hypothesis space is:

$$
\mathcal{M}_{\mathrm{reachable}}
=
\{\operatorname{decode}(z): z \in \mathcal{Z}_{\mathrm{frag}}\}
$$

If the target chemistry is outside $\mathcal{M}_{\mathrm{reachable}}$, model quality is capped by representation coverage before architecture matters.

## Evaluation Contract

Report generation metrics against the molecular object, not only the token string:

| Metric Family | Check |
| --- | --- |
| Validity | decoded molecule is graph-valid and chemically sane under the chosen state policy |
| Uniqueness | duplicate handling is done after standardization |
| Novelty | generated molecules are compared against training molecules and near-duplicate fragment recombinations |
| Diversity | scaffold, fingerprint, or fragment diversity is reported with the chosen representation |
| Utility | property, docking, or synthesis proxy is evaluated under an explicit benchmark protocol |

## Checks

- Is the fragment vocabulary fixed, learned, or dataset-specific?
- Does tokenization preserve stereochemistry, charge, isotopes, and aromaticity?
- Are attachment points unambiguous?
- Does validity guarantee imply chemically useful molecules or only graph-valid molecules?
- Is novelty measured against fragment recombination as well as full-molecule identity?
- Are standardization, scaffold split, and duplicate policy applied after decoding?
- Is the generated distribution evaluated against the same chemical-state policy used for training?

## Related Papers

- [[papers/generative-models/molexar|Molexar]]

## Related Concepts

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]

## Open Questions

- How does generation quality compare to atom-level SELFIES for property-conditioned tasks?
- Is there a standard fragment vocabulary, or is it task-dependent?
- Does fragment-level tokenization lose fine-grained control over individual atom placement?
