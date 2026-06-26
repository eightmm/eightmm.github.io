---
title: Sequence Identity Clustering
tags:
  - protein-modeling
  - evaluation
  - data
---

# Sequence Identity Clustering

Sequence identity clustering groups protein sequences so that highly similar sequences stay on the same side of a split. It is a core guardrail against leakage in protein modeling.

For two aligned sequences $s_i$ and $s_j$:

$$
\operatorname{identity}(s_i, s_j)
= \frac{1}{L}\sum_{\ell=1}^{L}
\mathbf{1}[s_{i,\ell}=s_{j,\ell}]
$$

$L$ is aligned length and $\mathbf{1}$ is an indicator function. A clustering threshold $\tau$ defines which sequences are too similar to separate:

$$
\operatorname{identity}(s_i, s_j) \ge \tau
$$

## Alignment Details

Identity depends on alignment policy. With gaps, a practical definition must state whether the denominator is full alignment length, matched positions, query coverage, or target coverage:

$$
\operatorname{coverage}(s_i,s_j)
=
\frac{\#\text{aligned non-gap positions}}
{\min(|s_i|,|s_j|)}
$$

A high identity over a tiny aligned fragment is not the same as high identity over a full domain.

## Cluster Split

After clustering, split assignment should happen at the cluster level:

$$
c(s_i)=c(s_j)
\Rightarrow
\operatorname{split}(s_i)
=
\operatorname{split}(s_j)
$$

where $c(s)$ is the sequence cluster ID. The cluster IDs, threshold, software, and input sequence normalization should be recorded because changing any of them changes the benchmark.

## Threshold Meaning

The threshold $\tau$ encodes the generalization claim:

- High threshold: removes near duplicates.
- Moderate threshold: tests family-level generalization.
- Low threshold: tests remote-homology or fold-level transfer.

There is no universal safe threshold. The right threshold depends on whether the claim is residue-level prediction, function prediction, pocket generalization, target-family transfer, or structure-based screening.

## Why It Matters

Random protein splits often place homologous or near-duplicate proteins in both train and test. The model then reports memorization or family lookup instead of generalization.

For protein-ligand tasks, sequence clustering alone may not be enough. If ligands also vary, a benchmark may need both protein cluster separation and ligand scaffold separation:

$$
\operatorname{split}(P,L)
=
g(c_{\mathrm{protein}}(P), c_{\mathrm{ligand}}(L))
$$

This avoids testing on a new row that is effectively a seen protein family with a seen ligand scaffold.

## Checks

- What identity threshold is used?
- Are chains, isoforms, fragments, and duplicate records normalized before clustering?
- Are all members of a cluster assigned to the same split?
- Does the benchmark need protein-family, target-family, or structure-level separation?
- Is the threshold public and reproducible?
- Is alignment coverage reported, not only percent identity?
- Are domain fragments, multi-chain complexes, and isoforms handled explicitly?
- For structure-based tasks, is ligand scaffold leakage checked separately?
- Are cluster IDs persisted so later runs use the same split?

## Related

- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/evaluation/leakage|Leakage]]
