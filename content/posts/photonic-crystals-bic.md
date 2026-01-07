---
title: "Trapping light in 3D Photonic Crystals"
date: 2024-11-20
draft: false
description: "On my condensed matter physics research"
tags: ["Physics", "Research"]
---

I've been working in the Photonics and Modern Electromagnetics Group under Professor Marin Soljačić at MIT since September 2023. This research project involved designing a 3D photonic crystal that could confine light without requiring a bandgap—a novel approach that challenges conventional wisdom about how to trap light. We're currently writing up the results for publication, and I presented this work at CLEO 2025! [Link to slides here](https://docs.google.com/presentation/d/1oXlbFHB7y2-UzHPJyz53l2HuTJFGhmYHzLizGtzFX_U/edit?usp=sharing)

## The problem: confining light in 3D PhCs

Photonic crystals are periodic structures made of materials with different refractive indices. Think of them as semiconductors for light instead of electrons. Just like how semiconductors have electronic bandgaps that prevent certain electron energies from propagating, photonic crystals can have photonic bandgaps that prevent certain light frequencies from propagating.

The conventional wisdom for trapping light in photonic crystals is simple: create a complete photonic bandgap. A bandgap is a range of frequencies where light simply cannot exist anywhere in the crystal. If you then introduce a defect, a local disruption of the periodic structure, you can create a cavity where light at those forbidden frequencies gets trapped. There are just no surrounding states for the light to couple to.

But creating a complete 3D photonic bandgap is *quite* hard. It requires strict geometric constraints, complex optimization, and structures that are often difficult or impossible to fabricate. The nonlinear optimization process to find these structures is computationally intensive, and there's no guarantee the result will be practical to manufacture.

So researchers have developed alternatives. **Index guiding** (the principle behind fiber optics) relies on total internal reflection and momentum conservation. **Anderson localization** introduces random disorder until light accidentally gets trapped, which is useful for random lasing, but unpredictable in where and how well light confines. **Bound states in the continuum (BICs)** engineer symmetry mismatches to prevent light from escaping, demonstrated in 2D photonic crystal slabs and nanocavities.

Each method has its place, but they all have limitations. This motivated our question: could we find a different mechanism for light confinement that doesn't rely on bandgaps in 3D photonic crystals?

## The key insight: vanishing density of states

Our approach exploits a subtle property of certain points in photonic band structures: **quadratic degeneracies** where multiple bands touch at a single frequency.

In 2D photonic crystals, researchers have successfully trapped light at **Dirac points**—linear degeneracies where two bands cross with conical dispersion. At a Dirac point, the photonic density of states (DOS) vanishes, meaning there are essentially no available bulk modes for light to escape into. By introducing a symmetry-breaking defect, you can create a localized mode at exactly this frequency. Since there are no bulk modes to couple to, the light stays trapped.

But what about 3D? We focused on **quadratic degeneracies** at high-symmetry points in the Brillouin zone. Unlike Dirac points with linear dispersion, these feature quadratic dispersion: $\omega(k) \approx \omega_0 + \alpha|k - R|^2$. The density of states near a quadratic degeneracy doesn't vanish entirely. It goes as $\sqrt{|\omega - \omega_0|}$, but it does go to zero at the degeneracy frequency $\omega_0$.

This creates an opportunity: if we can engineer a defect mode at exactly $\omega_0$, there are no bulk modes at that frequency for light to leak into.

## The design: symmetry as a design tool

We started with a 3D photonic crystal based on a simple cubic lattice with four silicon rods oriented along the body diagonals in each unit cell. This structure belongs to space group #224 (I$\bar{4}$3d), a non-symmorphic group that hosts a triply degenerate point at the R-point of the Brillouin zone at $(\pi/a, \pi/a, \pi/a)$.

The crucial detail is this three-fold degeneracy is protected by the crystal's symmetry group. It transforms as a three-dimensional irreducible representation (irrep) of the space group. As long as you maintain the full symmetry of the crystal, this degeneracy cannot be lifted, meaning it's robust against perturbations.

By carefully tuning the rod radius ($r/a = 0.18$) and using silicon's high refractive index ($\varepsilon = 12$), we isolated this degeneracy from nearby bands, ensuring it remained frequency-isolated. The normalized frequency came out to $\omega_D a/2\pi c \approx 0.323$.

In adition, we introduce a **spherical dielectric defect** at the center of the crystal. This breaks some of the crystal's symmetries while preserving others. Specifically, it reduces the local space group from #224 to #162. Crucially, space group #162 does *not* support three-dimensional irreps at the R-point.

This creates a **symmetry mismatch**: any defect mode at $\omega_0$ transforms under a one-dimensional or two-dimensional irrep of the reduced symmetry group. But the bulk states at the R-point transform under a three-dimensional irrep of the full symmetry group. These representations are orthogonal, meaning the defect mode mathematically cannot couple to the bulk modes.

In simpler terms, modes of different symmetries just don't mix. Even though they exist at the same frequency, they don't interact with each other.

## The simulations:

We validated this mechanism using finite-difference time-domain (FDTD) simulations with MEEP. The setup involved a $25a \times 25a \times 25a$ supercell with absorbing boundaries, initialized with a short electromagnetic pulse at the defect location.

Using Harminv, a signal processing tool that extracts resonant frequencies and quality factors from time-domain data, we tracked the defect modes. The results confirmed our predictions!

**1. The defect mode appears at the designed frequency**

When the spherical defect radius is tuned correctly, the defect mode sits right at $\omega_D = 0.323(2\pi c/a)$—exactly where the bulk density of states vanishes. At this spot, the quality factor Q reaches over $10^4$ in our modest system sizes.

**2. Q-factor scales exponentially with system size**

For the defect mode at $\omega = \omega_D$, the Q-factor increases exponentially as we make the system larger. For off-resonant modes (away from $\omega_D$), the scaling plateaus or becomes sub-exponential, which indicates they leak into bulk modes as expected.

**3. The field profile shows strong localization**

Looking at the electric field energy density $\varepsilon|\mathbf{E}|^2$, we see clear exponential decay away from the defect center. A 1D slice through the field shows a characteristic decay length of about $\lambda/a \approx 1.58$ lattice constants.

## What Makes This Different?

Our mechanism is fundamentally distinct from other confinement approaches:

- **Not a bandgap**: The photonic crystal has no complete bandgap. Bulk modes exist above and below $\omega_D$.

- **Not index guiding**: We're not relying on total internal reflection or momentum conservation—this is a point defect, not a waveguide.

- **Not Anderson localization**: The confinement is deterministic and engineered through symmetry, not random.

- **Different from 2D Dirac points**: Unlike 2D photonic crystals where the DOS vanishes linearly at Dirac points, our quadratic degeneracy has DOS that vanishes and then rises as $\sqrt{\omega}$. The sharp rise on either side of $\omega_D$ makes the spectral isolation particularly strong.

- **Different from 3D Weyl points**: Weyl points have linear degeneracies with a DoS that rises quadratically. Our quadratic degeneracy produces different physics: the square root scaling creates a different tradeoff between spectral isolation and mode density.

The exponential scaling we observe is particularly interesting. Typically, confinement at points of vanishing DoS shows quadratic scaling. The fact that we see exponential scaling suggests that the symmetry mismatch provides an additional layer of confinement beyond just the vanishing density of states. Future work could disentangle these contributions more carefully.

## What's next?
Early results suggest there are many other space groups where this approach could work. Each one potentially represents a new photonic crystal design that can trap light without a bandgap.

There are also practical questions to explore: How does fabrication disorder affect the symmetry protection? Can we extend this to longer wavelengths or different material systems? Could this enable new types of nanocavities or quantum emitter integration?

## A summary

This work demonstrates that the design space for photonic crystals is larger than traditionally assumed. Symmetry, some mathematical abstraction, is a design tool. By carefully breaking and preserving different symmetries, we can control how light behaves in ways that go beyond conventional mechanisms. There's some interesting group theory behind this that this blog didn't really go into, but maybe a later post! The ability to trap light in gapless structures could simplify the design and fabrication of photonic devices, opening new possibilities for integrated photonics, quantum optics, and light-matter interactions.