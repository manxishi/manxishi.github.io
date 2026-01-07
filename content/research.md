---
title: "Research"
layout: "research"
---

I'm working on a project on adaptive temperature scaling for constant-entropy LLM token generation for better long-form generation. Currently, people set a temperature parameter to control creativity vs determinism, but really what we care about is the entropy of the distribution tokens are being drawn from. Temperature is merely a proxy to this, and in most cases, it is a good proxy! However, we see entropy drift up or down at temperatures that deviate far from the normal range of 0.6-0.8(ish) as we go to long generation lengths. To combat this phenomenon, we propose to develop a per-token temperature formula that results in a constant per-token entropy. For cases where temperature is not a perfect proxy for entropy, this will potentially improve the quality of generation.

Previously, I worked on research in condensed matter physics, on 3d photonic crystals.
