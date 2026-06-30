---
title: Molecular Similarity
tags:
  - molecular-modeling
  - similarity
  - cheminformatics
---

# Molecular Similarity

Molecular similarity measures how close two molecules are under a chosen representation. It is used for retrieval, clustering, scaffold analysis, activity cliffs, nearest-neighbor baselines, and diversity checks.

For binary fingerprints $a,b\in\{0,1\}^d$, the Tanimoto similarity is:

$$
T(a,b)
=
\frac{a\cdot b}
{\lVert a\rVert_1+\lVert b\rVert_1-a\cdot b}
$$

The value ranges from 0 to 1 for binary fingerprints, with 1 meaning identical fingerprints.

## Key Ideas

- Similarity depends on representation: fingerprints, graphs, descriptors, conformers, and 3D shapes can disagree.
- High fingerprint similarity does not guarantee same binding pose or activity.
- Low similarity does not prove different mechanism if the representation misses relevant features.
- Similarity baselines are important before claiming a deep model learns new chemistry.

## Uses

- Retrieval: find nearest candidates to a query molecule.
- Clustering: define groups for split construction or diversity selection.
- Applicability domain: measure distance to known training chemistry.
- Activity cliffs: find high-similarity pairs with large label changes.
- Baselines: compare deep models against nearest-neighbor or fingerprint models.

For a nearest-neighbor baseline:

$$
\hat{y}(m)
=
y\left(\arg\max_{m_i\in \mathcal{D}_{\mathrm{train}}}
T(f(m), f(m_i))\right)
$$

where $f$ is the molecular representation and $T$ is a similarity function.

## Similarity Is Representation-Relative

Similarity should always be written as a function of representation:

$$
s(m_i,m_j)
=
\operatorname{sim}(f_\pi(m_i), f_\pi(m_j))
$$

where $\pi$ includes standardization, fingerprint settings, 3D conformer policy, or embedding model.

| Representation | Similarity captures | Misses |
| --- | --- | --- |
| fingerprint | local substructure overlap | 3D pose and target context |
| graph edit / MCS | explicit topology relation | pharmacophore similarity |
| descriptor vector | global physicochemical profile | substructure detail |
| conformer/shape | 3D volume and geometry | alternate conformers and protonation |
| learned embedding | task-conditioned relation | interpretability and domain shift |

Changing $f_\pi$ changes the meaning of the number, even if the metric name stays the same.

## Thresholds

A threshold is not portable across representations:

$$
\mathbb{1}[s(m_i,m_j)\ge \tau]
$$

The same $\tau$ can mean near-duplicate for one fingerprint and weak similarity for another embedding. State whether the threshold is used for retrieval, clustering, split construction, novelty filtering, or applicability-domain checks.

## Activity Cliff Boundary

Similarity can expose cases where smoothness assumptions fail:

$$
s(m_i,m_j)\ \text{high},
\quad
|y_i-y_j|\ \text{large}
$$

These pairs should be inspected separately because a model can have good average performance while failing on the chemistry that matters most.

## Practical Checks

- What representation defines similarity?
- Is similarity computed before or after molecule standardization?
- Are stereochemistry, protonation, tautomers, and salts handled consistently?
- Is the threshold chosen for retrieval, clustering, splitting, or diversity?
- Does a simple nearest-neighbor baseline explain the reported performance?
- Is similarity used as evidence of same mechanism, or only as a retrieval heuristic?
- Are activity cliffs reported separately from average nearest-neighbor behavior?

## Failure Modes

- Similarity is computed on raw molecules while splitting uses standardized molecules.
- A similarity threshold is reused across fingerprints, embeddings, and 3D shape without recalibration.
- High global similarity hides activity cliffs that dominate decision risk.
- A random split lets near-duplicate or congeneric molecules leak across train/test.
- Nearest-neighbor performance is not reported, so model novelty is unclear.

## Related

- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/substructure-search|Substructure search]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/tasks/retrieval|Retrieval]]
