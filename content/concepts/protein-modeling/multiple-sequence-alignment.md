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
=
\frac{1}{N}\sum_{n=1}^{N}
\mathbf{1}[A_{n,i}=a]
$$

With sequence weights $w_n$, a weighted profile is:

$$
p_{i,a}
=
\frac{\sum_{n=1}^{N} w_n\mathbf{1}[A_{n,i}=a]}
{\sum_{n=1}^{N} w_n}
$$

The effective number of sequences is often summarized as:

$$
N_{\mathrm{eff}}
=
\sum_{n=1}^{N} w_n
$$

This matters because a deep MSA and a shallow MSA do not provide the same amount of evolutionary evidence.

## Modeling Uses

- Conservation features for residue importance.
- Coevolution signals for contact or distance prediction.
- Homology context for structure prediction.
- Family-level grouping for split construction.

## MSA-Derived Signals

| Signal | Sketch | Interpretation |
|---|---|---|
| conservation | low column entropy | residue may be constrained |
| covariation | coupled residue changes | possible contact or functional coupling |
| insertion/deletion pattern | gap distribution | domain boundary or flexible region clue |
| MSA depth | $N_{\mathrm{eff}}$ | confidence in evolutionary statistics |

Column entropy is:

$$
H_i
=
-
\sum_{a\in\mathcal{V}}
p_{i,a}\log p_{i,a}
$$

Low entropy can reflect functional constraint, structural constraint, or dataset bias. It is not automatically a causal functional explanation.

## MSA vs Protein Language Model

An MSA explicitly retrieves homologous sequences. A [[concepts/protein-modeling/protein-language-model|Protein language model]] stores statistical regularities in parameters after pretraining.

| Method | Uses query-time homolog search? | Main risk |
|---|---:|---|
| MSA-based model | yes | database and template leakage |
| sequence-only language model | no | pretraining corpus overlap and family memorization |
| hybrid model | often | both risks apply |

## Leakage Risks

MSA generation can leak information when homolog or template databases include evaluation targets or close structural templates. The MSA database, search date, filtering threshold, and template policy should be treated as part of the evaluation protocol.

The data dependency is:

$$
\text{query sequence}
\xrightarrow{\text{search database at time }t}
\text{MSA}
\xrightarrow{\text{model}}
\text{prediction}
$$

If the search database contains benchmark-derived structures, target proteins, or near-identical homologs, the evaluation claim changes.

## Checks

- What database and search settings produced the MSA?
- Are low-quality, near-duplicate, or contaminated homologs filtered?
- Does MSA depth differ systematically between train and test targets?
- Does template or homolog retrieval leak benchmark structures?
- Are train/test proteins separated by sequence identity before MSA-derived features are interpreted?
- Is performance stratified by MSA depth or protein family?
- Is the model still usable when a target has few homologs?

## Related

- [[entities/sequence|Sequence]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-language-model|Protein language model]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
