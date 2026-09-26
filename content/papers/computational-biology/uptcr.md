---
title: UpTCR — Progressive Knowledge Transfer for Incomplete TCR–Antigen–HLA Interactions
aliases:
  - papers/uptcr
tags:
  - papers
  - computational-biology
  - protein-interaction
  - tcr
  - antigen
  - hla
  - transfer-learning
  - contrastive-learning
  - missing-modality
  - foundation-model
status: full-note
source_type: Journal
source_url: https://doi.org/10.1038/s41467-026-78075-x
---

# UpTCR: Progressive Knowledge Transfer for Incomplete TCR–Antigen–HLA Interactions

> **One-line takeaway:** UpTCR turns incomplete biomolecular interaction tuples into a transfer curriculum: learn entity and lower-order interaction modules first, then reuse them for the complete TCRα–TCRβ–peptide–HLA problem while softening uncertain negatives created by TCR cross-reactivity.

## Why this paper is worth saving

Biomolecular datasets rarely observe the same entity set. One source has protein–peptide pairs, another full complexes, another affinity labels, and another only a subset of partners. Complete-case training discards these partial observations; independent task models preserve the rows but fail to make lower-order interaction knowledge reusable.

For the complete recognition object

$$
\mathcal X=(\mathrm{TCR}_\alpha,\mathrm{TCR}_\beta,p,\mathrm{HLA}),
$$

the real data resemble

$$
\mathcal D
=
\mathcal D_{ph}\cup
\mathcal D_{bp}\cup
\mathcal D_{abp}\cup
\mathcal D_{bph}\cup
\mathcal D_{abph}.
$$

UpTCR uses this incompleteness as the learning schedule. That is directly relevant to unified biomolecular pretraining, where proteins, nucleic acids, ligands, structures and assay context are rarely present together.

A second durable lesson is label semantics. TCRs are cross-reactive, so an unobserved pair is not automatically a true negative. The paper therefore combines progressive transfer with soft contrastive learning.

---

## Metadata and artifacts

| Field | Value |
| --- | --- |
| Paper | UpTCR: a unified progressive knowledge transfer foundation model for robust T-cell receptor-antigen binding recognition |
| Authors | Tianxu Lv, Yang Xiao, Li Chen, Bing He, Maiyi Zhong, et al. |
| Journal | Nature Communications |
| Published | 2026-09-24 |
| DOI | [10.1038/s41467-026-78075-x](https://doi.org/10.1038/s41467-026-78075-x) |
| Official code | [tylerlv/UpTCR](https://github.com/tylerlv/UpTCR) |
| Code snapshot inspected | [cf042f45d2ece9e885b034c0ddca0124797d92a5](https://github.com/tylerlv/UpTCR/commit/cf042f45d2ece9e885b034c0ddca0124797d92a5) |
| Public weights | [Zenodo 10.5281/zenodo.20520000](https://doi.org/10.5281/zenodo.20520000) |
| Processed data | [Zenodo 10.5281/zenodo.15128399](https://doi.org/10.5281/zenodo.15128399) |
| Alternate data distribution | [DDDead/Uptcr_data](https://huggingface.co/datasets/DDDead/Uptcr_data) |
| Repository license | MIT |
| Article license | CC BY 4.0 |

> **Evidence boundary:** benchmark gains, low-data transfer, residue-level interpretation and the prospective melanoma results are author-reported. The public code, weights/data links, task scripts and implementation structure were independently inspected, but this note does not claim independent reproduction.

## Visual evidence

![UpTCR progressive knowledge-transfer pipeline](https://raw.githubusercontent.com/tylerlv/UpTCR/cf042f45d2ece9e885b034c0ddca0124797d92a5/pipeline.png)

*Source: official UpTCR repository, artifact pipeline.png, pinned at commit cf042f45d2ece9e885b034c0ddca0124797d92a5. The repository is MIT-licensed. The figure shows staged reuse of entity encoders and interaction-fusion modules; it is an author-produced architecture illustration, not independent performance evidence.*

The important point is that UpTCR is not mainly a missing-value imputer. It trains useful routes on the entity subsets that actually exist and transfers those components into richer tasks.

---

## 1. Progressive interaction-complexity transfer

Let an entity encoder be

$$
h_e=E_e(x_e),
$$

and an interaction module be

$$
h_{uv}=F_{uv}(h_u,h_v).
$$

The public implementation contains separate TCRα, TCRβ, epitope and MHC/HLA encoders, plus reusable TCRαβ and peptide–MHC fusion modules.

Conceptually,

$$
h_{\mathrm{TCR}}=F_{\mathrm{TCR}}(h_a,h_b),
\qquad
h_{\mathrm{pMHC}}=F_{\mathrm{pMHC}}(h_p,h_h),
$$

and the complete predictor uses

$$
\hat y=G(h_{\mathrm{TCR}},h_{\mathrm{pMHC}}).
$$

The complete-interaction trainer loads pretrained entity encoders and both fusion modules. Missing-modality trainers reuse the corresponding subsets. Thus the curriculum is more specific than “pretrain then fine-tune”: it progressively assembles learned interaction factors.

A joint multitask baseline could instead optimize

$$
\mathcal L_{\mathrm{joint}}
=
\sum_{t\in\mathcal T}\lambda_t\mathcal L_t.
$$

The scientific hypothesis is therefore that the ordering itself matters:

$$
\theta^{(1)}
\rightarrow
\theta^{(2)}
\rightarrow
\cdots
\rightarrow
\theta^{(K)}.
$$

This must be distinguished from the simpler explanation that progressive training merely consumes more data or more optimizer updates.

---

## 2. Representation and fusion

The current public code uses typed entity encoders rather than one universal concatenated-sequence model. TCRα and TCRβ combine full variable-region features, a CDR3-specific path, pretrained embeddings, residual 1D convolutions, self-attention and pooling. The epitope encoder similarly combines local sequence processing with pretrained embeddings; MHC has its own residual convolutional stack.

A simplified state is

$$
H_e^{(0)}
=
[\phi_{\mathrm{seq}}(x_e);\phi_{\mathrm{pre}}(x_e)],
$$

$$
H_e
=
\operatorname{FFN}
\left(
H_e^{(0)}
+
\operatorname{MHA}(H_e^{(0)})
\right).
$$

The public TCRABFusion and pMHCFusion modules concatenate residue states, apply residual convolutions, multi-head attention and feed-forward processing, then expose sequence and pooled interaction states.

The novelty is not standard attention. It is the **module boundary**: lower-order interaction representations remain reusable objects. Knowledge can therefore transfer at both entity and interaction levels.

This gives a clean ablation question: does performance come from better entity initialization, from learned interaction fusion, or from both?

---

## 3. Soft contrastive learning and false negatives

TCR cross-reactivity makes negative construction scientifically consequential. An unobserved TCR–peptide pair may simply be unmeasured.

A hard contrastive target assumes

$$
y_{ij}\in\{0,1\}.
$$

The article explicitly states that UpTCR uses soft contrastive learning to mitigate false negatives. A conceptual form is

$$
p_{ij}
=
\frac{\exp(s_{ij}/\tau)}
{\sum_k\exp(s_{ik}/\tau)},
$$

$$
\mathcal L_{\mathrm{soft}}
=
-\sum_jq_{ij}\log p_{ij},
$$

where the target distribution \(q_{ij}\) need not assign all unmatched pairs the same repulsive label.

This equation is explanatory rather than a claim of exact paper algebra. Exact reproduction should use the article/supplement and pinned code.

The reusable lesson is

$$
\text{unlabeled interaction}
\neq
\text{biological negative}.
$$

The same issue appears in protein–ligand screening, off-target modeling, PPI prediction and assay-mined weak labels.

---

## 4. Downstream and structural heads

The inspected specificity trainer uses binary cross-entropy:

$$
\mathcal L_{\mathrm{BCE}}
=
-
[y\log\hat y+(1-y)\log(1-\hat y)].
$$

It reports accuracy, ROC-AUC, AUPR, precision, recall and F1, and the inspected trainer selects the best validation-AUPR checkpoint.

The implementation also contains a structural head that predicts six residue-pair distance maps: TCRα–HLA, TCRβ–HLA, TCRα–TCRβ, peptide–TCRα, peptide–TCRβ and peptide–HLA.

For residue states \(H_A,H_B\),

$$
P_{ij}=\phi(h_i^A,h_j^B),
\qquad
\hat d_{ij}=g_\omega(P_{ij}).
$$

This provides a local interaction probe, but a distance map is not a full 3D complex. The evidence supports residue-level interaction/geometry prediction, not an AlphaFold-style co-folding claim.

---

## 5. Evidence and its boundary

The Nature Communications article reports stronger TCR-binding specificity and antigen–HLA affinity prediction than existing methods under its evaluated protocols, with emphasis on neoantigen settings. It also reports transfer to breast-cancer cohorts with limited data and pairwise residue-level interaction analysis across the tetramer.

The strongest prospective evidence is melanoma antigen-variant testing. The authors report **eight immunogenic peptides** eliciting T-cell responses and **one variant associated with immune escape**.

That is meaningful because a computational prioritization is followed by experimental immune-response evidence. It is not evidence of clinical efficacy:

$$
\text{T-cell response / immune-escape evidence}
\not\Rightarrow
\text{therapeutic benefit}.
$$

It also does not prove that the same architecture or curriculum is optimal for generic protein–ligand binding.

This review intentionally avoids table-specific benchmark numbers that were not independently extracted from the full paper/supplement. The durable mechanism and evidence scope are more important than one leaderboard delta.

---

## 6. OOD and split semantics

The repository provides separate few-shot and unseen checkpoints/scripts for complete and missing-modality configurations. But “unseen” is not a universal OOD condition.

| Holdout unit | Scientific question |
| --- | --- |
| new TCR / clonotype | receptor transfer |
| new peptide | antigen novelty |
| new HLA | presentation-context novelty |
| new TCR–peptide combination | combinatorial generalization |
| new cohort | population/domain shift |
| new assay/source | measurement shift |
| temporal holdout | future-data generalization |

A strong claim must name the actual split unit.

For protein–ligand modeling,

$$
\text{ligand scaffold OOD}
\neq
\text{protein-family OOD}
\neq
\text{pair OOD}
\neq
\text{assay/source OOD}.
$$

This is one of the most transferable evaluation lessons in the paper.

---

## 7. Reproducibility and artifact availability

The official repository exposes source code, pretrained individual/fusion weights, fine-tuned few-shot/unseen checkpoints for complete and incomplete settings, processed interaction and structure data, pretrained embeddings, training/prediction scripts, peptide–HLA affinity training and residue-pair structural prediction.

The README points to roughly 41 GiB of processed data and reports pretrained-embedding collections for 41,599 antigens, 27,066 TCRα sequences and 27,946 TCRβ sequences.

A reproducible rerun should pin:

- article/DOI version;
- Git commit;
- weight revision;
- processed-data revision;
- split construction;
- negative construction;
- pretrained embeddings;
- validation checkpoint rule;
- final metric implementation.

The repository inspected here is pinned to cf042f45d2ece9e885b034c0ddca0124797d92a5.

---

## 8. Main confounders

**More supervision versus better curriculum.** Progressive training sees lower-order data before the full task. A gain does not isolate curriculum order unless examples and optimization budget are matched.

**Pretrained-embedding capacity.** Some benefit may come from upstream sequence embeddings rather than interaction-level transfer.

**Negative-set provenance.** AUPR can change substantially with how unobserved pairs are converted into negatives.

**Missingness shortcut.** If each entity subset identifies a dataset/task, the pattern of missing entities itself can become predictive.

**Structural overclaim.** Pairwise distance prediction does not establish globally valid coordinate reconstruction.

**Domain specificity.** The biological evidence is TCR–peptide–HLA recognition; transfer to arbitrary biomolecular complexes remains a hypothesis.

---

## 9. Decision-useful ablations

The key experiment is a matched comparison:

1. full-complex only;
2. all partial + complete tasks jointly from scratch;
3. entity pretraining → full task;
4. pair interaction → triple interaction → full task.

Keep examples, total updates, architecture, downstream head and model-selection budget as matched as possible.

Then isolate transfer location:

- no transfer;
- entity encoders only;
- entity + pair fusion;
- entity + pair fusion + higher-order modules.

For negatives, compare trusted measured negatives, random unobserved hard negatives, similarity-aware hard negatives and soft targets. Report AUPR plus calibration and performance on independently measured negative sets.

Finally, evaluate a missing-context curve. For full entity set \(S\) and observed subset \(S'\),

$$
Q(S')=\text{task quality using only }S'.
$$

A reusable interaction model should degrade gracefully as context disappears.

---

## 10. Transfer to protein–ligand pretraining

The reusable abstraction is **partial-interaction curriculum**, not TCR-specific biology.

A protein–ligand corpus may contain ligand-only molecular data, protein-only sequence/structure data, activity pairs, pocket structures, protein–ligand complexes and assay-rich records.

A possible hierarchy is

$$
E_P,E_L
\rightarrow
F_{PL}^{\mathrm{weak}}
\rightarrow
F_{PL}^{\mathrm{structure}}
\rightarrow
F_{PL}^{\mathrm{task}}.
$$

The critical test is whether lower-order supervision improves the full interaction state without leaking information unavailable at deployment.

Required evaluation axes include ligand scaffold/similarity, protein family, protein–ligand pair, assay/source and time.

A useful extension is an explicit entity-availability mask:

$$
m_e\in\{0,1\},
$$

$$
z=
F_\theta\left(\{m_e,E_e(x_e)\}_{e\in\mathcal E}\right).
$$

Then missingness is a declared feature rather than a hidden task label. This is a follow-up hypothesis motivated by UpTCR, not a reported result.

---

## 11. Falsification

The strong progressive-transfer interpretation weakens if:

1. matched-data joint multitask training matches progressive training;
2. gains disappear after controlling pretrained-embedding capacity;
3. entity encoders explain nearly all gain and interaction-fusion transfer adds nothing;
4. soft negatives fail on independently measured non-binders or calibration;
5. unseen gains collapse under sequence-cluster, source or temporal holdouts;
6. missing-modality performance relies on missingness identifying the dataset;
7. residue-pair predictions fail on independently held-out structural complexes.

Surviving these controls would support a stronger conclusion: interaction factors were learned that remain useful across both entity availability and interaction complexity.

---

## Reproducibility checklist

| Item | Status |
| --- | --- |
| Peer-reviewed article | yes |
| Permanent DOI | yes |
| Open access | yes, CC BY 4.0 |
| Official repository | yes |
| Repository license | MIT |
| Pretrained weights | yes |
| Fine-tuned weights | yes |
| Processed data | yes |
| Missing-modality scripts | yes |
| Few-shot/unseen scripts | yes |
| Structural prediction head | yes |
| Independent reproduction here | no |
| Generic protein–ligand transfer evidence | not established |

---

## Related notes

- [[molecular-modeling/interactions|Interaction modeling]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[papers/computational-biology/chemical-dice-integrator|Chemical Dice Integrator]]
- [[papers/protein-modeling/multi-scale-antibody-binding|Multi-scale ML for Antibody-Antigen Binding]]

The comparison with Chemical Dice Integrator is useful. CDI asks whether rich training-time modalities can be distilled into a cheap deployment representation. UpTCR asks whether **partial interaction supervision can be ordered so that simpler interactions teach richer ones**.

---

## Final verdict

UpTCR should not be remembered primarily as a TCR leaderboard model. Its durable contribution is a learning contract for fragmented interaction data:

$$
\boxed{
\text{do not discard incomplete interaction tuples;}
\quad
\text{turn interaction completeness into a transfer curriculum}
}
$$

The public implementation makes that claim inspectable through typed entity encoders, TCRαβ and pMHC fusion modules, complete and missing-modality routes, few-shot/unseen checkpoints and residue-pair structural outputs.

The strongest caveat is that the transfer order is biologically structured for TCR–peptide–HLA recognition. A unified protein–ligand or all-biomolecule model should treat UpTCR as a **testable training principle**, not proof that the same hierarchy transfers automatically.

## Three durable takeaways

1. **Incomplete interaction records can be supervision rather than waste.** Lower-order interactions can initialize entity and fusion states for scarce higher-order tasks.
2. **Biological negatives need provenance.** Cross-reactivity makes “unobserved = negative” unsafe.
3. **OOD must name the held-out entity and relation.** Unseen receptor, peptide, combination, cohort, source and time are different claims.

## Sources

- Lv, T., Xiao, Y., Chen, L. et al. [UpTCR](https://doi.org/10.1038/s41467-026-78075-x). Nature Communications (2026), published 2026-09-24.
- Official implementation: [tylerlv/UpTCR](https://github.com/tylerlv/UpTCR), inspected at commit [cf042f45…](https://github.com/tylerlv/UpTCR/commit/cf042f45d2ece9e885b034c0ddca0124797d92a5).
- Official model weights: [Zenodo 10.5281/zenodo.20520000](https://doi.org/10.5281/zenodo.20520000).
- Official processed data: [Zenodo 10.5281/zenodo.15128399](https://doi.org/10.5281/zenodo.15128399) and [DDDead/Uptcr_data](https://huggingface.co/datasets/DDDead/Uptcr_data).
