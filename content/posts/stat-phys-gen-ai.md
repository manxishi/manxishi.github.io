---
title: "Statistical Physics and Generative AI"
date: 2025-11-04
draft: true
description: "On connections between statistical mechanics and modern generative AI"
tags: ["Physics", "Generative AI"]
---
The 2024 Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton for their work on neural networks and machine learning. While some debated whether this work truly belonged in "physics," the award highlighted something profound: that there was a deep connection between physics and artificial intelligence. Physics explains how the world works, and fundamentally, we want artificial intelligence to be able to perceive and understand the world like humans do. In statistical physics, the principles of energy minimization, phase transitions, and hiearchy is deeply relevant to generative AI, from transformers to diffusion models.

This post will trace the intellectual lineage from the early spin glass models in the 1980s to the diffusion models generating today's AI art and powering today's protein models, to name a few examples. We'll see how concepts like energy landscapes, Boltzmann distributions, and non-equilibrium thermodynamics provides not just the intuition behind, but also the fundamental mathematical frameworks for how some of these AI systems learn and generate.

## The Physics Foundations: Energy Landscapes and Memory

### Hopfield Networks as Statistical Mechanical Systems
In 1982, John Hopfield published a seminal paper that fundamentally reframed neural computation through the lens of physics. Rather than viewing neurons as logic gates, Hopfield treated them as interacting spins in a magnetic system, a direct analogy to the Ising model in statistical mechanics.

The key insight was defining an **energy function** for the network state:

$$E(\mathbf{s}) = -\frac{1}{2}\sum_{i,j} w_{ij} s_i s_j - \sum_i \theta_i s_i$$

where $\mathbf{s} = \{s_1, s_2, ..., s_N\}$ represents the binary states of neurons ($s_i \in \{-1, +1\}$), $w_{ij}$ are the synaptic weights connecting neurons, and $\theta_i$ are bias terms. This looks remarkably similar to the Hamiltonian of an Ising spin glass.

The network dynamics follow a simple rule: neurons update their states to reduce the total energy. When neuron $i$ updates:

$$s_i \leftarrow \text{sign}\left(\sum_j w_{ij}s_j + \theta_i\right)$$

This is a greedy descent on the energy landscape. Hopfield proved that the energy function acts as a **Lyapunov function**. It monotonically decreases with each update until the system reaches a local minimum (an attractor state). These attractor states serve as memories.

**The Storage Capacity Problem**: A natural question emerged: how many patterns can a Hopfield network store? Using statistical mechanics techniques, Amit, Gutfreund, and Sompolinsky (1985) showed that the capacity is approximately $C \approx 0.138N$ patterns for $N$ neurons, beyond which the network enters a "spin glass" phase where stored memories become confused.

This capacity limit comes from analyzing the network as a disordered magnetic system. When too many patterns are stored, the energy landscape becomes rugged with spurious local minima—phantom attractors that don't correspond to any stored pattern.

### From Deterministic to Stochastic: Boltzmann Machines

While Hopfield networks were deterministic, they had a problem: they could get stuck in poor local minima. Geoffrey Hinton and Terry Sejnowski (1983) introduced **stochasticity** through the Boltzmann machine, directly importing the Boltzmann distribution from statistical mechanics.

Instead of deterministically flipping neurons, Boltzmann machines update probabilistically:

$$P(s_i = 1) = \sigma\left(\frac{1}{T}\left(\sum_j w_{ij}s_j + \theta_i\right)\right)$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function and $T$ is the temperature parameter. At equilibrium, the probability distribution over network states follows the Boltzmann distribution!

$$P(\mathbf{s}) = \frac{1}{Z}e^{-E(\mathbf{s})/T}$$

where $Z = \sum_{\mathbf{s}} e^{-E(\mathbf{s})/T}$ is the partition function, a fundamental quantity in statistical mechanics that sums over all possible states of a system weighted by the Boltzmann factor.

**Temperature as a Control Parameter**: The temperature $T$ controls exploration vs. exploitation:
- High $T$: Random exploration, samples from nearly uniform distribution
- Low $T$: Exploitation, concentrates probability on low-energy states
- $T \to 0$: Deterministic, equivalent to Hopfield network
This is a familiar parameter nowadays in LLMs where temperature is a hyperparameter to control creativity vs randomness in generation.

### Restricted Boltzmann Machines: Making it Practical

The full Boltzmann machine is intractable to train because computing the partition function requires summing over all possible states. Paul Smolensky (1986) and later Hinton (2002) introduced **Restricted Boltzmann Machines (RBMs)** with a bipartite structure: visible units $\mathbf{v}$ and hidden units $\mathbf{h}$, with no connections within each layer.

The energy function simplifies to:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{a}^T\mathbf{v} - \mathbf{b}^T\mathbf{h} - \mathbf{v}^T\mathbf{W}\mathbf{h}$$

The bipartite structure enables efficient inference:
- $P(\mathbf{h}|\mathbf{v}) = \prod_i \sigma(b_i + \sum_j W_{ij}v_j)$ (all hidden units independent given visible)
- $P(\mathbf{v}|\mathbf{h}) = \prod_j \sigma(a_j + \sum_i W_{ij}h_i)$ (all visible units independent given hidden)

**Contrastive Divergence**: Training requires computing gradients of the log-likelihood, which involves intractable expectations over the model distribution. Hinton's breakthrough was **contrastive divergence (CD)**: approximate the negative phase by running only a few steps of Gibbs sampling from the data distribution, rather than waiting for equilibrium.

The gradient becomes:

$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} \approx \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

This is the difference between correlations in the data ("positive phase") and correlations under the model ("negative phase"), an interesting physical interpretation.

RBMs were stacked to form **Deep Belief Networks**, providing the first successful method for training deep neural networks (Hinton et al., 2006). While largely superseded by other methods, this work demonstrated that physics-inspired architectures could learn rich hierarchical representations.

## Modern Connections: Attention Mechanisms as Energy Minimization

The transformer architecture (Vaswani et al., 2017) revolutionized machine learning, yet its connection to energy-based models went largely unnoticed for years. Recent work has revealed that attention mechanisms are performing Hopfield-like energy minimization.

### Modern Hopfield Networks

Ramsauer et al. (2020) made a great discovery: the attention mechanism in transformers is mathematically equivalent to the update rule of a "modern" Hopfield network with exponential storage capacity.

Consider a Hopfield network with a modified energy function:

$$E(\mathbf{\xi}, \mathbf{X}) = -\text{lse}(\beta \mathbf{X}^T\mathbf{\xi}) + \frac{1}{2}\mathbf{\xi}^T\mathbf{\xi} + \frac{1}{2\beta}\log N + \text{const}$$

where $\text{lse}$ is the log-sum-exp function, $\mathbf{\xi}$ is the query state, $\mathbf{X}$ is a matrix of stored patterns (keys), and $\beta$ is an inverse temperature. The update rule that minimizes this energy is:

$$\mathbf{\xi}^{\text{new}} = \mathbf{X} \cdot \text{softmax}(\beta \mathbf{X}^T\mathbf{\xi})$$

This is exactly the attention mechanism! If we identify:
- Query: $\mathbf{q} = \mathbf{\xi}$
- Keys: $\mathbf{K} = \mathbf{X}$
- Values: $\mathbf{V} = \mathbf{X}$
- Temperature: $\beta = 1/\sqrt{d_k}$

Then attention computes:

$$\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \mathbf{V} \cdot \text{softmax}\left(\frac{\mathbf{K}^T\mathbf{q}}{\sqrt{d_k}}\right)$$

**Exponential Storage Capacity**: Unlike classical Hopfield networks with linear capacity ($C \propto N$), modern Hopfield networks have exponential capacity. The energy function uses the smooth log-sum-exp instead of quadratic interactions, allowing the network to store and retrieve exponentially many patterns.

This explains transformers' remarkable ability to store and retrieve information in-context. Each attention layer is performing associative memory retrieval. The query pattern retrieves similar key patterns from memory, weighted by their similarity (energy).

### Metastable States and Pattern Separation

The energy landscape perspective provides insight into how transformers process information. Each attention layer creates metastable states—patterns that are stable under the energy dynamics but not global minima. Multiple attention layers compose these energy landscapes, progressively refining representations.

Recent work by Bahri et al. (2021) analyzed transformers through the lens of dynamical systems theory, showing that attention layers can exhibit different dynamical regimes:
- **Ordered phase**: Small perturbations die out (stable, forgetting)
- **Chaotic phase**: Small perturbations amplify (unstable, sensitive)
- **Edge of chaos**: Delicate balance enabling rich computation

Operating near the edge of chaos may explain transformers' ability to generalize—they maintain information without being too rigid or too chaotic.

## Non-Equilibrium Thermodynamics: The Rise of Diffusion Models

While Hopfield networks and Boltzmann machines draw on equilibrium statistical mechanics, diffusion models are rooted in **non-equilibrium thermodynamics**.

### Thermalization and Data Generation

Sohl-Dickstein et al. (2015) introduced diffusion probabilistic models with an explicit connection to thermodynamics. The key insight: if we gradually add noise to data until it becomes pure Gaussian noise (thermalization), we can learn to reverse this process to generate new data.

The **forward diffusion process** is a Markov chain that gradually corrupts data over steps with some noise schedule.

<!-- $$q(\mathbf{x}_{t} | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_{t}; \sqrt{1-\beta_{t}}\mathbf{x}_{t-1}, \beta_{t}\mathbf{I})$$ -->

This is analogous to a physical system approaching thermal equilibrium.

The **reverse process** learns to denoise.

<!-- $$p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_{t}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_{t}, t))$$ -->

Training amounts to learning the reverse diffusion process is essentially learning to un-thermal equilibrium the data distribution from noise. This is inspired by Jarzynski's equality from non-equilibrium statistical mechanics, which relates forward and reverse processes through free energy differences.

### Score-Based Models and Langevin Dynamics

Song and Ermon (2019) provided an alternative view through **score-based models**. The score function is the gradient of the log probability density:

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

This score function points toward higher probability regions and pushes random samples toward data. If we can estimate the score function, we can generate samples using **Langevin dynamics**.

<!-- $$\mathbf{x}_{t+1} = \mathbf{x}_{t} + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_{t}) + \sqrt{2\epsilon}\mathbf{z}_{t}$$

where $\mathbf{z}_{t} \sim \mathcal{N}(0, \mathbf{I})$.  -->
The math comes from a discretized version of the overdamped Langevin equation from statistical mechanics, a stochastic differential equation describing particles in a potential field with friction and thermal noise.

The problem: the score function involves the unknown data distribution. The solution: train a neural network $\mathbf{s}_{\theta}(\mathbf{x})$ to approximate the score using **score matching**:

<!-- $$\mathcal{L}_{\text{SM}} = \mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_{\theta}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right\|^2 \right]$$ -->

<!-- Crucially, this objective can be computed without knowing $p(\mathbf{x})$ explicitly, using integration by parts (Hyvärinen, 2005). -->

**Connection to Diffusion**: Ho et al. (2020) showed that denoising diffusion probabilistic models (DDPM) are implicitly performing score matching. The denoising objective:

<!-- $$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_{t}, t) \right\|^2 \right]$$ -->

$$L_{\text{DDPM}} = E_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

is equivalent to learning the score function $$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$$ at different noise levels $t$.

### Stochastic Differential Equations Framework

Song et al. (2021) unified these perspectives using **stochastic differential equations (SDEs)**. The forward diffusion is a continuous-time SDE:

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

where $\mathbf{f}$ is a drift term, $g(t)$ is a diffusion coefficient, and $\mathbf{w}$ is a Wiener process. This is the **Itô SDE** describing how data diffuses into noise.

The reverse SDE (Anderson, 1982) is:

$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_{\mathbf{x}} \log p_t(\mathbf{x})]dt + g(t)d\bar{\mathbf{w}}$$

This reverse process runs time backward, requiring the score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at each time $t$. Once we've trained a neural network to estimate this score, we can generate samples by numerically integrating the reverse SDE.

**Probability Flow ODE**: Remarkably, there's a deterministic ODE with the same marginal distributions:

$$d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g^2(t)\nabla_{\mathbf{x}} \log p_t(\mathbf{x})]dt$$

This connects diffusion models to normalizing flows and provides an alternative sampling method.

### Statistical Mechanics of Diffusion Models

Recent work has analyzed diffusion models through the lens of statistical mechanics, revealing fundamental limits and phase transition phenomena.

**Thermodynamic Performance Limits**: Premkumar (2025) derived thermodynamic bounds on the performance of diffusion models using non-equilibrium statistical mechanics. The key insight: there's a fundamental tradeoff between the speed of generation and the accuracy (measured by KL divergence to the true distribution).

The minimum thermodynamic cost of generating a sample in time $T$ is bounded by:

$$W \geq k_B T \cdot D_{\text{KL}}(p_{\text{data}} \| p_{\text{model}}) / T$$

This is analogous to the Landauer limit in computation. There's a minimum possibly amount of energy required to erase one bit of information. Faster generation requires higher thermodynamic cost.

**Phase Transitions in Diffusion**: Several studies have identified phase transition phenomena in diffusion models. Li et al. (2025) showed that diffusion models exhibit a critical instability separating successful and failed generation. Near this critical point, the system shows:
- **Universality**: Critical exponents match mean-field universality classes
- **Symmetry breaking**: The system spontaneously selects one of many possible generated samples
- **Diverging correlation length**: Long-range correlations emerge near the critical point

This explains why diffusion models can generate diverse, coherent samples. They operate near a phase transition where the system is maximally sensitive to initial conditions while maintaining global structure.

**Entropy Production**: Ikeda et al. (2025) analyzed the entropy production rate during the reverse diffusion process, showing that faster sampling incurs higher entropy production cost. This establishes a **speed-accuracy tradeoff** from first principles of non-equilibrium thermodynamics:

$$\text{Error} \sim \frac{1}{\text{NFE} \cdot T}$$

where NFE is the number of function evaluations (sampling steps) and $T$ is effective temperature. This theoretical prediction matches empirical observations of diffusion models.

## Phase Transitions in Neural Networks

The connection between neural networks and statistical physics extends beyond specific architectures to fundamental properties of learning and computation.

### Critical Phenomena in Deep Networks

Recent work by Tamai et al. (2025) revealed that artificial neural networks exhibit phase transitions similar to those in statistical mechanics. They studied signal propagation in randomly initialized deep networks, finding distinct phases:

1. **Ordered phase** ($\sigma_w^2 < \sigma_w^{2*}$): Signals decay exponentially with depth
2. **Critical line** ($\sigma_w^2 = \sigma_w^{2*}$): Signals propagate without amplification or decay
3. **Chaotic phase** ($\sigma_w^2 > \sigma_w^{2*}$): Signals amplify exponentially with depth

Here $\sigma_w^2$ is the variance of weight initialization. At the critical point, networks exhibit:
- **Power-law correlations**: $C(L) \sim L^{-\alpha}$ where $L$ is depth
- **Universal scaling**: Behavior independent of specific details (width, activation)
- **Finite-size scaling**: Effects of finite width follow universal scaling relations

**Different Universality Classes**: Fascinatingly, different architectures belong to different universality classes, a concept in statistical physics that describes categories of phase transitions with the same critical exponents:
- **MLPs**: Mean-field universality class ($\alpha = 1$)
- **CNNs**: Directed percolation class ($\alpha = 0.159$)
- **ResNets**: Modified scaling due to skip connections

This suggests that architectural choices fundamentally change the physics of learning.

### Edge of Chaos Hypothesis

The optimal initialization for training lies near the critical point—the "edge of chaos." At this point:
- Gradients neither vanish nor explode
- Information flows efficiently through the network
- The network is maximally sensitive to inputs while maintaining stability

This connects to theories of computation in complex systems. Langton (1990) proposed that systems capable of universal computation exist near phase transitions between order and chaos. Neural networks may inherit this property.

### Implications for Architecture Design

These insights inform architecture design:
- **Normalization techniques** (BatchNorm, LayerNorm) dynamically push networks toward criticality
- **Skip connections** (ResNets) modify the critical surface, enabling training of very deep networks
- **Attention mechanisms** may operate near criticality, balancing stability and expressiveness

Understanding neural networks as critical phenomena provides principled ways to design and initialize architectures.

## Unified Perspective: What Physics Teaches Us

Across these different architectures and concepts, several unifying principles emerge from a physics perspective:

### Energy Minimization in Computation

From Hopfield networks to transformers to diffusion models, computation often reduces to energy minimization or sampling from probability-based or energy-based distributions. The energy function encodes:
- **Constraints**: What patterns are allowed (low energy)
- **Preferences**: Which patterns are preferred (lower energy)
- **Memory**: Stored patterns are attractor states (local minima)

This perspective suggests that good architectures have physically realistic energy landscapes with useful properties (few spurious minima, smooth gradients, etc).

### Temperature as a Universal Control

Temperature appears across models as a control parameter:
- **Boltzmann machines**: Controls sampling randomness
- **Softmax attention**: $\tau = \sqrt{d_k}$ controls sharpness
- **Diffusion models**: Noise schedule as temperature
- **Training dynamics**: Learning rate as temperature

High temperature enables exploration; low temperature enables exploitation. Annealing schedules (gradually decreasing temperature) are effective because they allow global exploration before local refinement.

### Phase Transitions and Emergence

Many phenomena in deep learning can be understood as phase transitions:
- **Grokking**: Sudden generalization as a phase transition in learning dynamics
- **Double descent**: Non-monotonic behavior near critical points
- **Emergence**: Sudden appearance of capabilities at scale
- **Memorization vs. generalization**: Competing phases in overparameterized models

Phase transitions are characterized by universality, and behavior depends only on a few key parameters and symmetries. This may explain why simple scaling laws predict complex system behavior.

### From Equilibrium to Non-Equilibrium

The field has progressed from equilibrium models (Hopfield, Boltzmann machines) to non-equilibrium models (diffusion, flow matching):
- **Equilibrium models**: Sample from Boltzmann distribution, require MCMC
- **Non-equilibrium models**: Transform distributions through deterministic/stochastic flows

Non-equilibrium models are often more practical because generation is a one-shot forward pass rather than iterative sampling to equilibrium. But they still obey thermodynamic constraints (speed-accuracy tradeoffs, entropy production bounds).

## Future Directions

There are a lot of interesting, possibly fruitful avenues of exploration in using physical intuition to design better architectures.

### Thermodynamically Efficient Architectures

Can we design architectures that minimize thermodynamic cost? Landauer's principle tells us that erasing information costs energy. Reversible computing (Bennett, 1973) suggests that computation can be made arbitrarily efficient by approaching reversibility. Recent work on reversible neural networks (Jacobsen et al., 2018) explores this direction.

Energy-based models and normalizing flows are exactly invertible because they never "erase" information. This suggests they may be fundamentally more efficient than irreversible alternatives.

### Better Understanding of Scaling

Why do larger models generalize better? Statistical mechanics suggests answers through:
- **Finite-size scaling**: Small models are far from thermodynamic limit
- **Renormalization group**: Scaling behavior determined by fixed points
- **Critical phenomena**: Operating near phase transitions exhibits universal scaling

A physics-based theory of scaling could predict when scaling will help and when it won't.

### Connections to Quantum Systems

Quantum computing offers new physics-inspired approaches:
- **Quantum Boltzmann machines**: Use quantum tunneling to escape local minima
- **Quantum annealing**: Hardware designed for optimization
- **Tensor networks**: Quantum-inspired representations for classical data

While practical quantum advantage remains elusive, the mathematical tools from quantum mechanics (tensor networks, entanglement measures) are proving useful for classical ML.

### Non-Equilibrium Training

Most training algorithms (SGD, Adam) are gradient descents with various heuristics. Could we design training dynamics inspired by non-equilibrium statistical mechanics?
- **Active matter**: Self-propelled particles that self-organize
- **Driven systems**: Systems far from equilibrium with energy injection
- **Critical phenomena**: Training dynamics near critical points

This could lead to more principled, physics-inspired optimizers.

## Conclusion

The 2024 Nobel Prize in Physics recognized Hopfield and Hinton not just for historical contributions, but for establishing a framework that continues to generate insights.

A summary of key connections:
- Attention mechanisms are modern Hopfield networks with exponential capacity
- Diffusion models perform thermalization and its reversal
- Neural network training exhibits phase transitions and critical phenomena
- Thermodynamic constraints impose fundamental limits on generation speed and accuracy
---

## References

- Hopfield, J. J. (1982). "Neural networks and physical systems with emergent collective computational abilities." *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.
- Hinton, G. E., & Sejnowski, T. J. (1983). "Analyzing cooperative computation." *Proceedings of the 5th Annual Conference of the Cognitive Science Society*.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). "Spin-glass models of neural networks." *Physical Review A*, 32(2), 1007.
- Smolensky, P. (1986). "Information processing in dynamical systems: Foundations of harmony theory." *Parallel Distributed Processing*, 1, 194-281.
- Hinton, G. E. (2002). "Training products of experts by minimizing contrastive divergence." *Neural Computation*, 14(8), 1771-1800.
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). "A fast learning algorithm for deep belief nets." *Neural Computation*, 18(7), 1527-1554.
- Fischer, A., & Igel, C. (2012). "An introduction to restricted Boltzmann machines." *Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications*, 14-36.
- Krotov, D., & Hopfield, J. J. (2016). "Dense associative memory for pattern recognition." *Advances in Neural Information Processing Systems*, 29.
- Ramsauer, H., Schäfl, B., Lehner, J., et al. (2020). "Hopfield networks is all you need." *arXiv preprint arXiv:2008.02217*.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention is all you need." *Advances in Neural Information Processing Systems*, 30.
- Bahri, Y., Dyer, E., Kaplan, J., et al. (2021). "Explaining neural scaling laws." *arXiv preprint arXiv:2102.06701*.
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). "Deep unsupervised learning using nonequilibrium thermodynamics." *International Conference on Machine Learning*, 2256-2265.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems*, 33, 6840-6851.
- Hyvärinen, A. (2005). "Estimation of non-normalized statistical models by score matching." *Journal of Machine Learning Research*, 6, 695-709.
- Song, Y., & Ermon, S. (2019). "Generative modeling by estimating gradients of the data distribution." *Advances in Neural Information Processing Systems*, 32.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., et al. (2021). "Score-based generative modeling through stochastic differential equations." *International Conference on Learning Representations*.
- Ikeda, S., Tanaka, H., & Yaida, S. (2025). "Speed-accuracy relations for diffusion models: Fundamental bounds from non-equilibrium thermodynamics." *Physical Review X* (in press).
- Premkumar, A. (2025). "Thermodynamic performance limits of diffusion models." *arXiv preprint*.
- Li, Z., Zhang, Y., & Wang, L. (2025). "The statistical thermodynamics of generative diffusion models: Phase transitions and universality." *Entropy*, 27(1).
- Tamai, K., Aoki, T., & Uid, T. (2025). "Universal scaling laws of absorbing phase transitions in artificial deep neural networks." *Physical Review Research*.
- Langton, C. G. (1990). "Computation at the edge of chaos." *Physica D*, 42(1-3), 12-37.
- Jarzynski, C. (1997). "Nonequilibrium equality for free energy differences." *Physical Review Letters*, 78(14), 2690.
- Anderson, B. D. (1982). "Reverse-time diffusion equation models." *Stochastic Processes and their Applications*, 12(3), 313-326.
- Bennett, C. H. (1973). "Logical reversibility of computation." *IBM Journal of Research and Development*, 17(6), 525-532.
- Jacobsen, J. H., Smeulders, A., & Oyallon, E. (2018). "i-RevNet: Deep invertible networks." *International Conference on Learning Representations*.