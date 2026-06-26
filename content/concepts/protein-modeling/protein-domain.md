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

Domain annotations can come from sequence families, structures, or functional databases:

$$
\phi_{\mathrm{domain}}(s,X)
\rightarrow
\{(a_j,b_j,c_j)\}_{j=1}^{k}
$$

where $c_j$ is a domain family, fold, or functional annotation.

## Key Ideas

- Domains help explain why sequence identity alone may not capture functional similarity.
- Domain boundaries may be annotated, predicted, or inferred from structure.
- Multi-domain proteins can have function and binding behavior that depends on domain arrangement.
- Splits by protein family or domain can be more realistic than random sequence splits.

## Why Domains Matter for Models

| Issue | Why it matters |
|---|---|
| boundary ambiguity | a residue window may cut through a functional unit |
| domain shuffling | global sequence similarity can miss shared domain modules |
| multi-domain context | function can depend on domain order and linker geometry |
| transfer claim | training and test sets may share domains even when full sequences differ |
| pocket context | ligand binding may involve one domain, an interface, or a conformational state |

For a domain-level prediction, the object is not the full protein:

$$
f_\theta(s_{a:b}, X_{a:b})
\neq
f_\theta(s_{1:L}, X_{1:L})
$$

The difference affects representation pooling, negative examples, and evaluation splits.

## Domain Splits

A domain-aware split should keep homologous domains together:

$$
\operatorname{domain\_family}(d_i)
=
\operatorname{domain\_family}(d_j)
\Rightarrow
\operatorname{split}(d_i)
=
\operatorname{split}(d_j)
$$

This is stricter than a random row split and often more relevant for protein function or structure-transfer claims.

## Practical Checks

- Are domain boundaries known or predicted?
- Does the model process full proteins, cropped domains, or residue windows?
- Are homologous domains separated across train and test?
- Does the task depend on domain-level structure, global arrangement, or local residues?
- Are missing domains or low-confidence regions handled explicitly?
- Are domain annotations from the same database/version across train and test?
- Are linker regions and inter-domain contacts included or discarded?

## Related

- [[entities/protein|Protein]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/binding-site|Binding site]]
