---
title: "Physics of Language"
date: 2025-11-04
draft: false
description: "On interesting things I'm learning about language models during my research"
tags: ["LLMs", "Research"]
---

Recently, I came across Zeyuan Allen-Zhu's [physics of language models tutorial at ICML 2024](https://physics.allen-zhu.com/), and I thought it was quite interesting. It offers insights into probiling the theoretical capabilities of language models and suggests practical guidance on how we can improve these models.

Inspired, I thought some more about my own research on temperature scaling tokens, and this is a dump of my thoughts before I forget.

More on the intuition behind why we think having generations with constant per-token marginal entropy over generated length will be better. Right now, language model outputs deteriorate as it produces many thousands of tokens. It's bad at long form generation because usually the model either exhibits exploding entropy (too high, degenerate gibberish output) or collapsing entropy (too low, deterministic and repetitive output).

At early steps, a moderate temperature may produce well-calibrated entropy. But as the internal state diverges, the model’s true uncertainty (entropy of the base logit distribution) changes, but you’re applying a fixed global scaling T. The model accumulates tiny biases from attention/norms over long sequences, which affects the logit distribution even though the applied temperature stays constant, which is what causes entropy drift. When people set a temperature, they are really hoping to set the entropy of the marginal distribution, or the diversity/determinism of samples generated.

To be more specific, marginal entropy refers to the entropy over model draws (not conditional distribution entropy per sample). That’s the expected entropy of the predictive marginal integrating over the prefix distribution itself. Computing it exactly would require infinitely many samples, but in practice, we estimate it empirically over a batch of generation runs (30 at the moment). 

A clear distinction to make is between conditional (per-sample) entropy and marginal entropy over all samples.
Conditional entropy is

$$\mathbb{E_{\textit{b}}}[H[p_b]] = \frac{1}{B} \sum_{b=1}^{B} \sum_{v} (-p(v) \log p(v))$$

It describes how uncertain the model is about the next token, within a *single* sample trajectory. It measures per-sample confidence, not diversity of output. Marginal entropy is

$$H[\mathbb{E_{\textit{b}}}[p_b]] = H(\bar{p}) = -\sum_{v} \bar{p}(v)\log \bar{p}(v), \quad \text{where} \quad \bar{p}(v) = \frac{1}{B}\sum_{b=1}^{B} p_b(v)$$

It measures how uncertain the model is across all possible generations and represents diversity/agreement across different samples. Marginal entropy is the number we care about in this use case in improving outputs of long-form generation. Conditional entropy can still look normal (higher at the beginning of a sequence, when a model is still exploring, and stabilizing to a lower value later), while marginal entropy explodes or collapses. For the marginal entropy exploding case, think of it as each trajectory can be confident (low conditional entropy) in *different* directions.

Currently, to find a temperature schedule that achieves a constant marginal enrtopy over token position, I use scipy optimize to minimize a loss that I define to be MSE on first 300 out of 1000 tokens, comparing the actual marginal entropy vs the target marginal entropy. Loss over only the beginning tokens because thats the part that struggles the most with keeping a constant marginal entropy.

How do we make this more computationally tractible?
Marginal entropy is what we want, but there could be some proxy to this. Some logits statistics shortcut can use variance as a proxy to the marginal entropy over the full vocabulary.
It would be a lot easier to do online updates, like control-theory esque updates to the temperature, based on some observed entropy vs target entropy. However, note that this is no longer marginal entropy if we are only looking at a singular sequence. Though we could generate a few in a batch and call it an approximation of marginal entropy. But that's not very accurate.
