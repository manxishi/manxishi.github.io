---
title: "Notes on Power Sampling"
date: 2026-02-03
draft: false
description: "From a recent paper"
tags: ["LLMs", "Research"]
---

Here I talk about the paper ["Sampling: Your Base Model is Smarter than You Think"](https://arxiv.org/abs/2510.14901) [1]. The authors argue that RL-posttraining methods effectively sharpen the base model's existing distribution, and they can obtain similar or better results **at inference time** without training by smartly sampling a **power distribution** $p^\alpha$. It's similar in a way to using a low temperature setting, where more probable tokens are more likely to be picked and less probable tokens are almost never picked. However, a key distinction is dealing with probabilities at the token level (low temperature) vs. the sequence level (power sampling). **More formally, low-temperature sampling rescales conditional probabilities** $p(x_t \mid x_{<t})$, **while power sampling rescales the joint sequence probability** $p(x_{0:T})$.

This paper proposes to consider sequence-level probabilities, trying to avoid the problem of high-probability next tokens that could provide a dead-end answer later, which would have a low total sequence-level probability. The target distribution is

$$
p_\alpha(x_{0:T}) \propto p(x_{0:T})^\alpha,
$$

where $\alpha \ge 1$ controls the degree of sharpening. This is interesting because there is the idea of "pivotal tokens", tokens in a chain of thought that shift the direction of the answer, and if you avoid choosing the wrong token at that point, it could greatly improve the answer as a whole. **The paper argues that power sampling favors tokens with fewer but higher-likelihood future completions, rather than tokens with many low-likelihood futures.**

The algorithm is similar to Metropolis–Hastings (MCMC), since calculating the total probability of all possible future sequences is intractable. Rather than explicitly marginalizing over all suffixes, the method uses relative likelihood ratios to accept or reject local resamplings. This paper has a sort of "block-wise" structure, where they divide the generated output into blocks of size $B$, and revise-then-freeze the blocks sequentially. This narrows the search space and edits to make. Within each block, they repeatedly pick a random position, resample the remainder of the block, and accept the proposal with a Metropolis–Hastings acceptance probability.

They perform iterations of "pick a point, resample, and compare" only within that block. Once the iterations are done, this block is considered frozen, and they move on to the next block. Allowing the model to revise at any point, the inference cost would grow exponentially. Keeping the sequential edit-and-freeze structure keeps the added cost linear in sequence length (up to a constant factor depending on the number of MCMC steps).

There is a parameter $\alpha$, the inverse of temperature, that describes the acceptance behavior of a revision. At the two extremes, $\alpha = 1.0$ samples from the base model directly, while $\alpha \to \infty$ is like deterministically accepting any resampled sequence that strictly increases the likelihood **under the base model.** Another important parameter is the number of MCMC steps to take within a block, which naturally is the parameter that affects inference-time compute and acts as a form of test-time scaling.

This paper seems to depend on the autoregressive nature of LLMs and that reasoning is a chain: if a model can find the most confident path for the first step (the first block), it is more likely to succeed in the next step (the second block), etc. It's essentially like a greedy search at the block level rather than the token level, though still stochastic within each block due to MCMC resampling. The core intuition, as the authors state, is derived from the notion of distribution sharpening, **with the key claim being that RL sharpens existing high–likelihood reasoning paths rather than creating fundamentally new ones.**

---

**References**

[1] [Sampling: Your Base Model is Smarter than You Think](https://arxiv.org/abs/2510.14901). arXiv:2510.14901.
