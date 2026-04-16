---
title: "On-Policy Self-Distillation"
date: 2026-04-15
draft: false
description: "On a recent paper"
tags: ["LLMs", "Research"]
---

Here I talk about the paper ["Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"](https://arxiv.org/abs/2601.18734) [1]. The authors propose a training method called OPSD where a single LLM plays the role of both teacher and student, using ground-truth solutions as privileged information to provide dense, token-level supervision. The key intuition is that evaluation is easier than generation: if you show a model the correct answer before asking it to judge a student's attempt, it can provide much richer feedback than a binary right or wrong reward signal.

The setup works by creating two views of the same model with different prompts. The student policy $p_S(\cdot \mid x)$ sees only the problem $x$. The teacher policy $p_T(\cdot \mid x, y^\star)$ sees both the problem and a reference solution $y^\star$. Crucially, these are the same model with the same weights, the only difference is the conditioning context. The student generates an on-policy rollout $\hat{y} \sim p_S(\cdot \mid x)$, and then both policies evaluate this rollout token by token. The training objective minimizes the per-token divergence between the teacher and student distributions along the student's own trajectory.

This is interesting because it sidesteps several problems at once. Compared to standard SFT, you avoid distribution mismatch: the student trains on its own outputs, not on some expert's trajectories. Compared to GRPO/RLVR, you get dense token-level feedback instead of a single binary reward for the entire sequence. And compared to on-policy distillation with a separate teacher, you don't need a larger external model.

The teacher prompt is designed so the model first sees the reference solution, then is asked to "solve the problem using your own approach." The teacher never actually generates tokens, the rationalization happens implicitly through a single forward pass (prefilling). This is the "learning by understanding solutions" idea: just as a student can review a correct answer and retrace why it works, a capable LLM can condition on the answer to produce a better next-token distribution, which then serves as supervision.

OPSD matches or exceeds GRPO on competition-level math benchmarks (AIME 2024/2025, HMMT 2025) across Qwen3-1.7B/4B/8B, while using only 1 rollout per problem with 1024 tokens, versus GRPO's 8 rollouts of 16k tokens each. GRPO stagnates quickly because of reward diversity collapse, when all samples in a group get the same binary reward, the gradient is zero. OPSD avoids this entirely since its learning signal comes from distribution matching, not outcome rewards.

The main limitation is that the method depends on the model being capable enough to "rationalize" the given solution. If a problem is beyond the model's comprehension, the teacher policy can't provide meaningful supervision even with the answer in hand. The authors suggest curriculum learning as a potential fix, gradually increasing difficulty as the model improves, but leave this for future work.

---

**References**

[1] [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734). arXiv:2601.18734.