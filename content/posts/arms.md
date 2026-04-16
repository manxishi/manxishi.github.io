---
title: "Are LLMs secretly EBMs"
date: 2026-02-26
draft: false
description: "On a recent paper"
tags: ["LLMs", "Research"]
---

Here I talk about the paper ["Autoregressive Language Models are Secretly Energy-Based Models:
Insights into the Lookahead Capabilities of Next-Token Prediction"](https://arxiv.org/abs/2512.15605) [1].

We are all very familiar with ARMs as the dominant paradigm for LLMs, but not as many are familiar with energy-based models (EBMs). They are less prevalant despite their ablity to naturally characterize optimal post-training alignment mostly because they are extremely computationally costly. In theory, EBMs are arguably more capable language models. They possess the intrinsic ability to reason ahead of the current token because they model sequence-level, not token-level, distributions. Sampling from these unnormalized distributions is costly and often requires Markov-chain Monte-Carlo (MCMC) methods. But interestingly, it can be derived that MaxEnt RL has an analytical solution that is exactly an EBM. This paper shows that ARMs are approximate EBMs and provides mathematical derivations for this bijection as well as error bounds error bounds for distilling EBMs into ARMs. There is a lot of math to read through, but the main takeaway is pretty clear and interesting!

---

**References**

[1] [Autoregressive Language Models are Secretly Energy-Based Models:
Insights into the Lookahead Capabilities of Next-Token Prediction](https://arxiv.org/pdf/2512.15605). arXiv:2512.15605.
