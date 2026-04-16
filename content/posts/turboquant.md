---
title: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
date: 2026-03-28
draft: false
description: "On a recent paper"
tags: ["LLMs", "Research"]
---

Here I talk about the paper ["TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"](https://arxiv.org/pdf/2504.19874) [1].

Inference hardware and ML efficiency are important and interesting topics. This recent paper introduces a near-optimal quantization scheme that cuts the KV cache by at least 6x. It has applications in vector search as well, not surprisingly. This information-theoretic problem of compression traces back to Claude Shannon's seminal work on Source Coding Theory, which is now known as vector quantization. This fundamental problem has implications on modern decoder-based transformer architectures, which during inference are often memory bottlenecked because of accessing a large KV cache.

The core idea of TurboQuant is elegant: apply a random rotation to the input vectors, which causes each coordinate to follow a concentrated Beta distribution (converging to Gaussian in high dimensions via concentration of measure). The key insight is that after rotation, distinct coordinates become nearly independent, which means you can just apply optimal scalar quantizers (Lloyd-Max) to each coordinate independently and there is no need to reason about correlations between dimensions. This reduces vector quantization to a set of solved 1d problems.

There's a subtle issue though: quantizers optimized for MSE are biased for inner product estimation. This matters because attention in transformers is computed via inner products between queries and keys. To fix this, TurboQuant uses a two-stage approach: first apply an MSE-optimal quantizer, then apply a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual. The result is an unbiased inner product estimator with low distortion.

The algorithm is also **data-oblivious** (it doesn't need to see your data distribution ahead of time) and **online** (you can quantize vectors as they arrive, which is exactly what you need for KV cache quantization during autoregressive decoding). The codebooks are precomputed using the Max-Lloyd algorithm on Beta distributions, so at runtime it's just a rotation followed by table lookups.

The authors also prove information-theoretic lower bounds on the best achievable distortion for any vector quantizer, and show that TurboQuant is within a factor of $\approx 2.7$ of optimal. Experimentally, they achieve quality-neutral KV cache compression at 3.5 bits per channel, and only marginal degradation at 2.5 bits. Game-changing or not I don't know, but it is definitely cool and did drop the stocks of some memory hardware makers!

---

**References**

[1] [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/pdf/2504.19874). arXiv:2504.19874.


