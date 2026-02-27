---
title: "Research"
layout: "research"
---

<!-- I'm working on a project on adaptive temperature scaling for constant-entropy LLM token generation for better long-form generation. Currently, people set a temperature parameter to control creativity vs determinism, but really what we care about is the entropy of the distribution tokens are being drawn from. Temperature is merely a proxy to this, and in most cases, it is a good proxy! However, we see entropy drift up or down at temperatures that deviate far from the normal range of 0.6-0.8(ish) as we go to long generation lengths. To combat this phenomenon, we propose to develop a per-token temperature formula that results in a constant per-token entropy. For cases where temperature is not a perfect proxy for entropy, this will potentially improve the quality of generation.
 -->
I'm working on a project on distilling a power distribution into an LLM for better reasoning without traditional post-training RL methods or expensive MCMC sampling. Based on the recent paper "Reasoning with Sampling: Your Base Model is Smarter than you Think", we see that power sampling can improve the capabilities of LLMs, nearing and sometimes exceeding performance with GRPO. The idea is that we want these performance gains without the costly increase in inference time compute with MCMC, so we are trying to distill this knowledge into the base model.

Previously, I worked on research in condensed matter physics, on 3d photonic crystals.