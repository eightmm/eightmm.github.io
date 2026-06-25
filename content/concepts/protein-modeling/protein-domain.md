---
title: Protein Domain
tags:
  - protein-modeling
  - sequence
  - structure
---

# Protein Domain

A protein domain is a compact sequence and structure unit that can often fold, evolve, or function semi-independently. Domains are useful for interpreting protein families, transfer learning, and generalization splits.

For a protein sequence $s_{1:L}$, a domain can be represented as an interval:

$$
d = (a,b),
\qquad
1 \le a \le b \le L
$$

For multi-domain proteins, the representation becomes a set of intervals or structural regions:

$$
D = \{d_1,\ldots,d_k\}
$$

## Key Ideas

- Domains help explain why sequence identity alone may not capture functional similarity.
- Domain boundaries may be annotated, predicted, or inferred from structure.
- Multi-domain proteins can have function and binding behavior that depends on domain arrangement.
- Splits by protein family or domain can be more realistic than random sequence splits.

## Practical Checks

- Are domain boundaries known or predicted?
- Does the model process full proteins, cropped domains, or residue windows?
- Are homologous domains separated across train and test?
- Does the task depend on domain-level structure, global arrangement, or local residues?
- Are missing domains or low-confidence regions handled explicitly?

## Related

- [[entities/protein|Protein]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
