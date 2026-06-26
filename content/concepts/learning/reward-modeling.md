---
title: Reward Modeling
tags:
  - reward-modeling
  - preference-optimization
  - evaluation
---

# Reward Modeling

Reward modeling learns a scoring function that approximates human, expert, user, or verifier preferences. It is often used when the true objective is difficult to write as a direct loss.

For pairwise preference data $(x,y_w,y_l)$, a common model is:

$$
P(y_w \succ y_l \mid x)
=
\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
$$

The reward model is trained with:

$$
\mathcal{L}_{\mathrm{RM}}
=
-\log \sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
$$

where $y_w$ is preferred over $y_l$.

## Key Ideas

- A reward model turns qualitative judgment into an optimization target.
- The model can rank generated answers, agent trajectories, protein designs, molecular candidates, or tool-use outcomes if the comparison protocol is well defined.
- Reward models are proxies, so they can be exploited by policies trained too aggressively against them.
- Reward modeling should be paired with held-out human review, task metrics, or external verification.

## Reward Model Contract

| Field | Question |
| --- | --- |
| Scored object | final answer, trajectory, action, code diff, molecule, pose, design, or experiment plan? |
| Conditioning context | prompt, task, target, pocket, assay, environment, or verifier state? |
| Training label | pairwise preference, scalar score, pass/fail, or rubric item? |
| Aggregation | one reward, multi-objective reward, or rubric-weighted reward? |
| Calibration | are reward scores comparable across tasks or only within a prompt? |
| Held-out check | what evaluates the reward model itself? |

The score:

$$
r_\theta(x,y)
$$

is a learned proxy. The note should state what real-world or benchmark objective it is supposed to approximate.

## Failure Modes

| Failure | Symptom |
| --- | --- |
| reward hacking | policy finds high-reward outputs that humans or verifiers reject |
| distribution shift | reward model fails on outputs from a newer policy |
| length/style bias | reward tracks verbosity, formatting, or surface style |
| annotator bias | preference reflects rater artifacts rather than task quality |
| sparse coverage | reward model uncertain on rare task types |
| proxy mismatch | high reward does not improve external metric |

For agent trajectories, reward should distinguish correct final answer from unsafe or inefficient tool use.

## Evaluation

| Evaluation | What It Tests |
| --- | --- |
| preference accuracy | whether reward ranks held-out pairs correctly |
| calibration by score bucket | whether higher scores mean better external quality |
| adversarial examples | whether reward can be gamed |
| policy optimization audit | whether optimizing reward improves independent metrics |
| slice analysis | whether reward works across task types and domains |

## Practical Checks

- What exactly is being compared: final answer, full trajectory, code diff, molecule, pose, or experimental plan?
- Are preference labels consistent across annotators and task types?
- Does the reward model generalize outside the data distribution it was trained on?
- Are high-reward outputs also high quality under external evaluation?
- Is the reward model evaluated on outputs from the optimized policy, not only the data-collection policy?
- Are uncertainty, disagreement, or low-confidence reward regions handled?

## Related

- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
