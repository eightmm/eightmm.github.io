---
title: Multiple Sequence Alignment
tags:
  - protein-modeling
  - sequence
---

# Multiple Sequence Alignment

A multiple sequence alignment (MSA) aligns homologous sequences so that columns approximate evolutionary correspondence. In protein modeling, MSAs provide conservation and coevolution signals.

An MSA can be represented as a matrix:

$$
A\in\mathcal{V}^{N\times L}
$$

where $N$ is the number of homologous sequences, $L$ is alignment length, and $\mathcal{V}$ is the residue vocabulary plus gap symbols.

A column frequency profile is:

$$
p_{i,a}
= \frac{1}{N}\sum_{n=1}^{N}
\mathbf{1}[A_{n,i}=a]
$$

## Modeling Uses

- Conservation features for residue importance.
- Coevolution signals for contact or distance prediction.
- Homology context for structure prediction.
- Family-level grouping for split construction.

## Leakage Risks

MSA generation can leak information when homolog or template databases include evaluation targets or close structural templates. The MSA database, search date, filtering threshold, and template policy should be treated as part of the evaluation protocol.

## Checks

- What database and search settings produced the MSA?
- Are low-quality, near-duplicate, or contaminated homologs filtered?
- Does MSA depth differ systematically between train and test targets?
- Does template or homolog retrieval leak benchmark structures?
- Are train/test proteins separated by sequence identity before MSA-derived features are interpreted?

## Related

- [[entities/sequence|Sequence]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[research/protein-modeling/index|Protein modeling]]
