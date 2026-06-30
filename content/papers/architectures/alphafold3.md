---
title: AlphaFold3
aliases:
  - papers/alphafold3
  - papers/accurate-structure-prediction-of-biomolecular-interactions-with-alphafold-3
  - papers/protein-modeling/alphafold3
  - papers/sbdd/alphafold3
tags:
  - papers
  - architectures
  - protein-modeling
  - structure-prediction
  - diffusion-model
  - biomolecular-interactions
---

# AlphaFold3

> The paper changes AlphaFold from a protein-structure predictor into a unified biomolecular complex predictor with an atom-level diffusion architecture.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Accurate structure prediction of biomolecular interactions with AlphaFold 3 |
| Authors | Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, Sebastian W Bodenstein, David A Evans, Chia-Chun Hung, Michael O'Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvilė Žemgulytė, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B Fuchs, Hannah Gladman, Rishub Jain, Yousuf A Khan, Caroline M R Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula, Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D Zhong, Michal Zielinski, Augustin Žídek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, John M Jumper |
| Year | 2024 |
| Venue | Nature |
| Paper | [Nature](https://www.nature.com/articles/s41586-024-07487-w) |
| DeepMind page | [AlphaFold](https://deepmind.google/science/alphafold/) |
| Code | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| Status | full note started |

## One-Line Takeaway

AlphaFold3 replaces the AlphaFold2-style structure module with a diffusion-based coordinate generator and expands the input/output contract from protein chains to biomolecular complexes containing proteins, nucleic acids, small molecules, ions, and modified residues.

## Question

AlphaFold2 answered a protein monomer structure prediction problem:

$$
\text{protein sequence}
\rightarrow
\text{protein structure}.
$$

Many biological and drug-discovery questions are not monomer problems. They ask for a joint structure:

$$
\text{proteins}
+
\text{DNA/RNA}
+
\text{ligands}
+
\text{ions}
+
\text{modifications}
\rightarrow
\text{complex geometry}.
$$

The architecture question:

> Can one model represent mixed biomolecular entities and generate their joint atom-level coordinates?

The paper's answer:

$$
\text{unified atom representation}
+
\text{pair reasoning}
+
\text{diffusion coordinate generation}
\Rightarrow
\text{joint biomolecular structure prediction}.
$$

## Main Claim

The narrowed architecture claim is:

$$
\text{entity-aware biomolecular input}
+
\text{token/pair trunk}
+
\text{atom-level diffusion module}
\Rightarrow
\text{accurate complex prediction across molecular types}.
$$

This is different from a docking-only claim. A docking pipeline often assumes:

$$
\text{fixed receptor}
+
\text{ligand conformer/search}
\rightarrow
\text{pose}.
$$

AlphaFold3 instead predicts the joint complex:

$$
\hat{X}_{\text{complex}}
=
f_\theta(\mathcal{E}),
$$

where $\mathcal{E}$ is a set of molecular entities and $\hat{X}_{\text{complex}}$ is the predicted coordinate set.

## Architecture Contract

| Component | Role |
| --- | --- |
| entity input | proteins, nucleic acids, ligands, ions, modified residues |
| atom/token features | represent mixed molecular objects |
| pair representation | stores relationships between residues, atoms, or tokens |
| trunk | reasons over biomolecular context before coordinate generation |
| diffusion module | denoises atom coordinates into a complex structure |
| confidence heads | estimate reliability and interaction quality |
| sampling/refinement | produces candidate structures through generative denoising |

The output is atom-level geometry:

$$
\hat{X}
\in
\mathbb{R}^{N_{\text{atoms}}\times 3}.
$$

For any global rigid transform:

$$
X' = RX+t,
$$

the structural claim should be unchanged. Complex prediction therefore has the same rigid-transform symmetry issue as protein structure prediction, but with more heterogeneous objects.

## Why It Is an Architecture Paper

AlphaFold3 should be read as an architecture paper because it changes the modeling contract:

| Axis | AlphaFold2 | AlphaFold3 |
| --- | --- | --- |
| primary target | protein structure | biomolecular complex structure |
| output route | structure module with residue frames | diffusion-based atom coordinate generation |
| molecular scope | mainly proteins, then multimer variants | proteins, nucleic acids, ligands, ions, modified residues |
| downstream relevance | protein modeling | protein modeling, interaction modeling, SBDD |
| core architectural shift | Evoformer plus structure module | trunk plus diffusion module |

This is not merely "AlphaFold2 with more input types." The generative coordinate module changes how structure is produced.

## Input Object Contract

A biomolecular complex can be represented as:

$$
\mathcal{C}
=
(\mathcal{E}, A, B, F),
$$

where:

- $\mathcal{E}$ is the set of molecular entities;
- $A$ is the set of atoms;
- $B$ stores bonds or connectivity;
- $F$ stores features such as residue type, atom type, chain, entity type, charge, or modification.

For proteins:

$$
e_{\text{protein}}
=
(s_1,\ldots,s_L).
$$

For ligands:

$$
e_{\text{ligand}}
=
(G_{\text{mol}}, \text{chemistry features}),
$$

where $G_{\text{mol}}$ is a molecular graph.

For nucleic acids:

$$
e_{\text{NA}}
=
(b_1,\ldots,b_L),
$$

where $b_i$ is a nucleotide token.

The hard part is not just concatenating inputs. The model must preserve molecular identity, connectivity, symmetry, and interaction context.

## Unified Coordinate Prediction

AlphaFold3 predicts a joint coordinate set:

$$
\hat{X}
=
\{\hat{x}_1,\ldots,\hat{x}_N\},
\qquad
\hat{x}_i\in\mathbb{R}^3.
$$

The coordinate set includes atoms from multiple entities. A protein-ligand complex is not:

$$
\hat{X}_{\text{protein}}
\text{ first, then dock ligand}.
$$

It is closer to:

$$
(\hat{X}_{\text{protein}},\hat{X}_{\text{ligand}})
=
f_\theta(\text{protein},\text{ligand},\text{context}).
$$

This matters because interactions can affect relative placement and local conformations.

## Diffusion Module

Diffusion models learn to recover clean data from noise. For coordinates:

$$
X_t
=
\alpha_t X_0
+
\sigma_t \epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

The denoising model predicts either the clean coordinates, the noise, or a related update:

$$
\epsilon_\theta(X_t,t,c),
$$

where $c$ is conditioning information from the trunk.

A reverse denoising step can be written abstractly as:

$$
X_{t-1}
=
g_\theta(X_t,t,c).
$$

For AlphaFold3 reading, the important contract is:

$$
\text{noisy atom cloud}
+
\text{biomolecular context}
\rightarrow
\text{less noisy atom coordinates}.
$$

The model generates a structure through iterative coordinate refinement rather than a single deterministic residue-frame output.

## Why Diffusion Helps Here

Biomolecular complexes have multi-modal uncertainty:

- ligands can bind in different poses;
- side chains can rearrange;
- loops can move;
- complexes can have ambiguous interfaces;
- multiple conformations may be plausible;
- input context can be incomplete.

Diffusion gives an architecture for conditional structure generation:

$$
p_\theta(X \mid \mathcal{E})
$$

rather than only a point estimate:

$$
\hat{X}=f_\theta(\mathcal{E}).
$$

In practice, evaluation still often reports best or ranked predicted structures. The paper should not be read as solving conformational ensemble modeling completely.

## Token and Atom Representations

Mixed biomolecular modeling needs at least two levels of representation:

| Level | Examples | Why Needed |
| --- | --- | --- |
| token/entity level | residues, ligand tokens, nucleotides | long-range context and sequence/entity structure |
| atom level | atom coordinates and features | final geometry, contacts, pose, chemistry |
| pair level | token-token or atom-atom relations | interactions, distances, interface reasoning |

A generic pair representation:

$$
z_{ij}
\in
\mathbb{R}^{d_z}
$$

stores a relation between object $i$ and object $j$.

For interaction prediction, pair features are central:

$$
\text{binding mode}
\approx
\text{consistent cross-entity geometry}.
$$

## Pair Reasoning

A biomolecular complex can be treated as a dense relational object:

$$
G=(V,E),
\qquad
E=V\times V.
$$

Pair state:

$$
z_{ij}
\leftrightarrow
\text{relation between object } i \text{ and } j.
$$

For a protein-ligand complex, cross-entity pairs are crucial:

$$
z_{i,a}^{\text{protein-ligand}}
$$

where $i$ is a protein residue/atom and $a$ is a ligand atom/token.

The architecture must distinguish:

| Relation | Meaning |
| --- | --- |
| protein-protein | interface, multimer packing |
| protein-ligand | pocket pose and contacts |
| protein-DNA/RNA | sequence-specific binding, backbone geometry |
| ligand internal | molecular graph and conformer validity |
| ion/modified residue | coordination and local chemistry |

## Relation to AlphaFold2

AlphaFold2:

$$
s,M,T
\rightarrow
\text{MSA/pair trunk}
\rightarrow
\text{structure module}
\rightarrow
\hat{X}_{\text{protein}}.
$$

AlphaFold3:

$$
\mathcal{E}
\rightarrow
\text{biomolecular trunk}
\rightarrow
\text{diffusion module}
\rightarrow
\hat{X}_{\text{complex}}.
$$

The key shift:

$$
\text{protein-specific coordinate prediction}
\rightarrow
\text{general biomolecular coordinate generation}.
$$

## Relation to Docking

Classical docking often separates receptor preparation, ligand conformer generation, search, and scoring:

$$
\text{receptor}
+
\text{ligand}
\rightarrow
\text{pose candidates}
\rightarrow
\text{score/rank}.
$$

AlphaFold3 is closer to joint prediction:

$$
\text{protein}
+
\text{ligand}
\rightarrow
\text{complex structure}.
$$

This distinction matters:

| Docking View | AlphaFold3 View |
| --- | --- |
| receptor may be fixed | complex can be predicted jointly |
| ligand pose search explicit | coordinate generation learned |
| scoring function central | model confidence and learned structure distribution central |
| usually ligand-focused | all biomolecular entities share a prediction framework |

But AlphaFold3 should not be treated as replacing every docking workflow. Docking tasks often involve screening millions of ligands, protonation/tautomer states, induced fit, water networks, affinity ranking, and assay-specific validation.

## Relation to Diffusion Models

Generic diffusion:

$$
x_T \sim \mathcal{N}(0,I),
\qquad
x_{t-1}=g_\theta(x_t,t,c).
$$

AlphaFold3-style structure diffusion:

$$
X_T \sim \text{noisy atom coordinates},
\qquad
X_{t-1}=g_\theta(X_t,t,\mathcal{E}).
$$

The conditioning object is scientific:

$$
c
=
\text{molecular identity, connectivity, templates, sequence, pair context}.
$$

The output is constrained by chemistry and geometry, not just image realism.

## Relation to Geometric Deep Learning

Coordinate prediction should respect rigid transforms:

$$
X' = RX+t.
$$

If the input coordinate frame is rotated and translated, coordinate outputs should transform consistently:

$$
f(RX+t)=Rf(X)+t.
$$

Scalar confidence or interaction probabilities should be invariant:

$$
c(RX+t)=c(X).
$$

AlphaFold3 is a major example of generative geometric modeling for scientific structures.

## Evidence Reading

The paper evaluates many interaction categories. The evidence should be separated by claim:

| Evidence Type | Supports | Does Not Prove Alone |
| --- | --- | --- |
| protein-ligand benchmarks | improved pose prediction for some drug-like interactions | reliable affinity ranking |
| protein-protein/nucleic-acid complexes | broader biomolecular complex modeling | all biological assemblies are solved |
| comparison to specialized tools | one model can compete across categories | every category is equally strong |
| confidence estimates | some reliability signal | all hallucinations are detected |
| PoseBusters-style evaluation | physical plausibility checks | experimental binding or activity |

The paper is strong because it broadens the target distribution. That also makes evaluation harder: each entity type and interaction type has different failure modes.

## Benchmark Boundary

For SBDD, distinguish:

$$
\text{pose accuracy}
\neq
\text{binding affinity}
\neq
\text{virtual screening enrichment}
\neq
\text{prospective drug discovery success}.
$$

AlphaFold3 can improve pose or complex prediction while still leaving:

- affinity calibration;
- activity cliffs;
- assay transfer;
- ligand protonation;
- water/metal coordination;
- induced fit;
- conformational ensembles;
- prospective validation.

This boundary is important for public notes because AlphaFold3 attracts broad claims that can easily become too strong.

## Implementation and Reproducibility Reading

When reading or trying to use AlphaFold3-like systems, separate:

| Question | Why It Matters |
| --- | --- |
| Are model weights available? | determines reproducibility and local deployment |
| Is inference code available? | determines implementation auditability |
| Are inputs standardized? | ligands, CCD codes, SMILES, modifications, templates |
| Are multiple samples generated? | diffusion outputs may vary |
| Are confidence scores calibrated? | ranking and filtering depend on reliability |
| Is the benchmark leakage-controlled? | PDB time splits and homolog/template leakage matter |

This is not just an architecture issue. For scientific modeling, access, input format, and evaluation protocol are part of the claim contract.

## Failure Modes

| Failure Mode | Why It Matters |
| --- | --- |
| treating a plausible generated complex as experimental truth | generative models can produce convincing but wrong structures |
| using predicted pose as affinity | pose and binding energy are different claims |
| ignoring ligand chemistry state | protonation, tautomer, stereochemistry, and charge affect binding |
| ignoring water and cofactors | missing context can change the interface |
| comparing against docking without matching inputs | some methods may use different receptor/template information |
| mixing server output and paper architecture claims | access route and model version may differ |
| assuming all molecular types are equally solved | performance varies by interaction category |

## Common Misreadings

### "AlphaFold3 solves docking."

No. It makes a major advance in complex structure prediction, including protein-ligand interactions, but docking workflows include screening, affinity ranking, chemical state handling, and prospective validation.

### "Diffusion means it models all conformational ensembles."

No. Diffusion is a generative architecture, but practical prediction may still produce limited samples and can miss biologically relevant states.

### "One model for many molecule types means one evaluation metric is enough."

No. Protein-protein interfaces, ligand poses, nucleic-acid complexes, and ion coordination need different checks.

### "High confidence means the interaction is biologically real."

No. A predicted complex can be geometrically plausible while missing biological context, concentration, cellular state, or assay evidence.

## Later-Paper Checklist

When reading post-AlphaFold3 papers, check:

- Does the model predict joint complexes or dock into fixed structures?
- Are proteins, ligands, nucleic acids, ions, and modifications represented in one input schema?
- Is the coordinate generator diffusion-based, flow-based, or deterministic?
- Are molecular graph constraints enforced or learned?
- Are ligand protonation, tautomer, stereochemistry, and charge explicit?
- Are templates, homologs, and PDB date splits controlled?
- Are pose, affinity, and screening claims separated?
- Is confidence calibrated per interaction type?
- Are multiple samples generated and ranked?
- Is there prospective or held-out temporal validation?

## Why It Matters

AlphaFold3 is a major architecture transition:

$$
\text{structure prediction}
\rightarrow
\text{interaction structure generation}.
$$

For this wiki, it connects:

$$
\text{AlphaFold2}
\rightarrow
\text{diffusion models}
\rightarrow
\text{protein-ligand modeling}
\rightarrow
\text{structure-based drug discovery}.
$$

It is exactly the kind of paper that should be cross-listed: architecture, computational biology, protein modeling, and SBDD all need it, but the canonical note can live once in `papers/architectures`.

## Limitations

AlphaFold3 should be treated as a powerful structure predictor, not a complete biological oracle.

Important limitations:

- generated structures still need experimental or orthogonal validation;
- binding affinity is not identical to pose quality;
- ligand chemistry state can dominate real-world behavior;
- water, ions, cofactors, and missing context can matter;
- flexible and disordered regions remain difficult;
- model access and reproducibility constraints affect scientific use;
- benchmark performance may not transfer to every target class.

The defensible claim:

$$
\text{AlphaFold3}
\Rightarrow
\text{major advance in joint biomolecular structure prediction}.
$$

The overclaim to avoid:

$$
\text{AlphaFold3}
\Rightarrow
\text{drug discovery solved}.
$$

## Connections

- [[papers/architectures/alphafold2|AlphaFold2]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[papers/protein-modeling/index|Protein modeling papers]]
- [[papers/sbdd/index|Structure-Based Modeling Papers]]
- [[papers/computational-biology/index|Computational Biology papers]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/molecular-modeling/protein-ligand-representation-contract|Protein-ligand representation contract]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor-ligand preparation]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[papers/architectures/index|Architecture papers]]
