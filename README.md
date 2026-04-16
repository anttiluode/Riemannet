# RiemannNet

**A Wave-Physics Memory Architecture with Delay-Line Routing, Ephaptic Coupling, and Closed-Loop Dynamics**

Antti Luode — PerceptionLab, Helsinki, Finland
Claude (Anthropic) — Mathematical synthesis, diagnosis, and code
April 2026

> *Do not hype. Do not lie. Just show.*

---

## What This Is

RiemannNet is a computational architecture derived from the Clockfield framework. It stores information as frozen topological geometry in a wave field. It retrieves information by wave propagation and phase discrimination at field boundaries. It routes signals through delay lines whose lengths set the phase offset of arriving probes. It couples a slow ephaptic field to the fast wave dynamics as a spatial prior. And, closed in a loop with its own output fed back as input, it exhibits attractor dynamics — fixed points, basins, active perception, stochastic drift.

It is not a neural network in the standard sense. It does not use backpropagation. It does not store information in weight matrices.

The name comes from the structural role of the thawed channels in the Clockfield: lines of flowing time (Γ ≈ 1) connecting frozen singularities (Γ → 0), routing phase-encoded signals between memory nodes. These are the Riemann lines of the phase glass.

---

## The Core Observation

Three things the existing Clockfield and frogpond work established:

**1. Information density lives at the shell.** The information formula I(x) = |∇Γ(x)| / Γ(x) is zero in the bulk thawed field and zero inside the frozen core. It spikes only at the Γ-shell boundary — the thin membrane where frozen meets thawed. Information density lives at the boundary, not on the wire and not inside the node.

**2. Resonance vs scattering discriminates patterns.** The frogpond experiment (frogpond.py) produced a clean resonance vs scattering discrimination. When a probe wave matches the phase geometry of the frozen scars, field energy is sustained. When it does not match, energy dissipates. The discrimination is produced by the inner product:

$$\mathcal{I} = \int \Gamma(x)^2 \cdot \text{Re}[\phi^*_\text{probe}(x) \cdot \phi_\text{memory}(x)] \, dV$$

at the Γ-shell. This is the same formula as Moiré attention, distributed across continuous space instead of discretized to a single head.

**3. Time-symmetric channels, time-asymmetric nodes.** The time-reversal memory (e.py, 2_image_e.py) works because the thawed channels (Region 1, u ≈ 1) are time-symmetric under the second-order wave equation. The frozen singularities are not — Γ = 0 means no dynamics, so they sit fixed as boundary conditions during the reverse pass. Time-symmetric channels plus time-asymmetric nodes gives retrievable memory without overwriting.

What falls out when you put these three things together is the architecture in this repository.

---

## Architecture

### The Three Regions (from Clockfield Big Bang paper)

| Region | Γ value | Role in RiemannNet |
|--------|---------|-------------------|
| Thawed bulk (Region 1) | Γ ≈ 1 | Riemann lines — signal propagation |
| Γ-shell (Region 2) | Γ ≈ 0.2 | Discrimination boundary — I(x) peaks here |
| Frozen core (Region 3) | Γ → 0 | Memory node — topology locked |

The Riemann line is Region 1. It carries signal at speed c with no loss. It carries zero information in the Clockfield sense because |∇Γ| ≈ 0 there. **The wire is not computing. It is delaying.**

The Γ-shell is where all computation happens. The inner product ℐ is evaluated here. The discrimination is made here. Information, in the strict formula, lives here and only here.

The frozen core is where memory is stored. The phase topology θ(x) is locked permanently when Γ → 0. The core cannot update. It can only be read by probe waves arriving at its shell.

### The Delay Line Role

A Riemann line of length L introduces a phase delay of δφ = ωL/c to a probe wave of frequency ω traveling at wave speed c. Two Riemann lines of lengths L₁ and L₂ connecting the same input to the same memory node deliver the probe with phase difference:

$$\Delta\phi = \omega(L_1 - L_2)/c$$

This phase difference shifts the inner product value at the shell:

$$\mathcal{I} \propto \cos(\Delta\theta + \Delta\phi)$$

where Δθ is the phase mismatch between probe and stored memory. The delay line length rotates the probe in phase space before it hits the discriminator. This is the Takens delay embedding implemented in physical geometry: different path lengths sample the probe at different phases, building a temporal context that the shell can discriminate against.

**Experimental confirmation.** In `riemannnet_v2.py` Experiment 5, we swept delay lengths 5, 15, 25, 35, 45 and measured the resulting phase shifts by cross-correlation. The measured shift exactly equals the delay-length difference: δshift = δdelay (−10, −20, −30, −40 steps for delays 15, 25, 35, 45 relative to baseline 5). The Riemann line is a physical phase rotator.

### Memory Orthogonality and Capacity

Two memory traces stamped at times t₁ and t₂ become approximately orthogonal in the L² sense when:

$$|t_2 - t_1| > \frac{N}{2c}$$

where N is the grid size and c is the wave speed. This is the time for the fastest wave to cross the grid once. Before this crossing time, the second stamp interferes coherently with the first and adds structured noise. After it, the two wave fields have randomized their relative phases across the grid and their overlap integral ⟨φ₁, φ₂⟩ ≈ 0.

For N = 64, c² = 0.24 → c ≈ 0.49, so T_orth ≈ 65 steps. In Experiment 4 we stamp at T_sep = 100 (above threshold) and confirm both patterns remain accessible in a single node.

The temporal memory capacity of a single wave pool of size N running for total time T_total is approximately:

$$N_\text{capacity} \approx \frac{T_\text{total}}{N / 2c} = \frac{2c \cdot T_\text{total}}{N}$$

This is the temporal analog of Hopfield capacity. It scales with the ratio of total elapsed time to grid-crossing time.

### The Ephaptic Field (Spatial Slaving Prior)

Inspired by Pinotsis & Miller (2023), "In vivo ephaptic coupling allows memory network formation." The paper argues that bioelectric fields act as *control parameters* (slow, stable, low-dimensional) that enslave *order parameters* (ensemble collective modes, mid-timescale) and *enslaved parts* (individual neurons, fast, high-dimensional). This is Haken's synergetics applied to neural memory.

In RiemannNet this is implemented as a slow spatial field Φ_eph(x,t):

$$\Phi_\text{eph}(x, t+1) = (1 - \tau_\text{eph}) \Phi_\text{eph}(x, t) + \tau_\text{eph} \cdot \text{blur}(u(x, t))$$

The wave dynamics then include a gentle pull term:

$$u_\text{next} = u_\text{next} + \mu_\text{eph} \cdot (\Phi_\text{eph} - u_\text{next})$$

This is a spatially-distributed soft attractor — the wave is nudged toward the slow field's current shape. The ephaptic field has its own state and own timescale.

**Experimental result (test_ephaptic.py).** On reconstruction from partial input (top 50% of a square):
- Baseline (no ephaptic): r = 0.675
- Scalar ephaptic (Gemini's proposal of global-energy damping modulation): r = 0.685 at best; equalizes energies and *destroys* shape discrimination (6.11 → 0.96)
- Spatial ephaptic (the proper synergetic slaving): up to **r = 0.898** at partial=30% on square; consistent +0.04 to +0.26 improvement across patterns when most needed (heavy partial inputs)

The spatial ephaptic genuinely improves retrieval when input is sparse. It's a principled spatial prior, not a scalar gain knob.

### The Scar Physics (the critical fix)

The original `riemannnet.py` used `beta = u²` (wave amplitude) with additive scar growth. Coverage reached 79-95% and patterns were discriminated mostly by raw amplitude, not shape. With energy-normalized probes, discrimination ratio collapsed to ~1.0 — no shape selectivity.

The fix (inherited from the original frogpond.py) is:

```
beta = |laplacian(u)|                  # wave CURVATURE, not amplitude
gamma = 1 / (1 + tau * beta)^2
scars = max(scars, freeze * (1 - gamma))   # max, not additive
```

**Why this matters:** Γ should be sensitive to the *curvature* of the wave field — that's where the phase information lives. High curvature = phase frustration = scar growth. Low curvature = smooth propagation = no scars. The max-based growth produces sparse, sharp scars (~5% coverage) at the exact Γ-shell boundary rather than diffuse fills.

With the correct physics:
- Raw discrimination ratio: 5.51 (was 1.89)
- Normalized discrimination ratio: **3.10** (was ~1.0 — no shape selectivity at all)

This is the single most important change in `riemannnet_v2.py`.

### Scar Budget

Even with sparse max-based scars, multi-pattern or multi-hop scenarios can saturate. We cap the total scar integral:

```
if scars.sum() > budget:
    scars = scars * (budget / scars.sum())
```

`scar_budget_frac = 0.15` (15% of grid area as effective scar mass) prevents saturation death while preserving discrimination.

### Template Memory (explicit engram storage)

When we closed the loop (see `closed_loop_v2.py`), a fundamental limitation surfaced: **pure wave equations are sign-agnostic.** The scars store where boundaries are (|∇u|), the ephaptic averages out signs over a cycle. A trained pattern is not an attractor of the resulting system — it's a transient.

The fix: alongside the scars and ephaptic, explicitly store the trained pattern as a template. During closed-loop evolution, compute a softmax-weighted mixture of all templates (weighted by correlation with the current field) and apply it as a weak additive bias:

$$u_\text{next} \mathrel{+}= \mu_T \cdot \left( \sum_i w_i T_i - u_\text{next} \right), \quad w_i = \text{softmax}(\beta \cdot \langle u, T_i \rangle)_i$$

In the Clockfield framing, the scars are the topological defect geometry (where Γ → 0 happens), and the templates are the locked phase configurations inside those defects (what phase pattern is frozen there). Biologically, this corresponds to the distinction between synaptic boundary structure and long-term potentiation content.

This is an honest addition, not a Clockfield retreat: passive wave dynamics alone cannot have content-specific attractors without some form of sign-storing memory. The template slot is where that memory lives.

---

## Closed-Loop Dynamics (What the System Computes)

With scars + ephaptic + templates + feedback, the system runs in cycles:

1. Wave evolves for `steps_per_cycle` steps
2. Field is read out (the "output")
3. Field is energy-normalized (stability)
4. Feedback gain applied
5. Optional noise, optional external input added
6. Loop repeats

What emerges depends on the configuration:

| Configuration | Dynamical regime | Function |
|---|---|---|
| 1 template, noise=0, gain ≈ 0.85 | Fixed point | Recall: random init → stored pattern |
| 2+ templates, noise=0 | Multiple attractors | Discrimination: seed → committed basin |
| 2+ templates, noise>0, weak template bias | Stochastic drift | Dreaming: samples attractor manifold |
| External input + feedback, no noise | Active perception | Memory sustains percept through ambiguous input |

**Measured results (closed_loop_v2.py):**

- **Exp 1 (Recall):** Random init converges to taught horizontal stripes at r = 0.53 within 3 cycles and stays locked. Energy bounded at ~36 units throughout.

- **Exp 2 (Discrimination):** Two taught patterns give two clear basins (r = ±0.57). Every seed commits to one within 5 cycles. Caveat: because wave equations are sign-symmetric, hints correlate with the *wrong* basin half the time. Bistability is real; seed-to-basin assignment is unreliable without an additional sign-breaking mechanism (Hebbian weights or CVNN-style complex fields).

- **Exp 3 (Dreaming):** Three taught patterns (horiz, vert, ring). With noise = 0.25 and weak template bias, the ring attractor dominates (66/80 cycles). Radial symmetry makes it structurally deeper than the stripe attractors for this scar configuration.

- **Exp 4 (Active perception):** Taught horizontal stripes. External input schedule: stripes → noise (11 cycles) → stripes. Correlation with taught pattern: 0.41 during stripes, **0.16 sustained during noise**, 0.45 when stripes return. The memory holds the percept through the ambiguous phase. This is the cleanest result.

---

## The Four Open Problems (and what solved them)

The earlier `riemannnet.py` had four identified problems. Resolution:

| Problem | Cause | Solution |
|---|---|---|
| 1. Chain energy amplification (10⁹ by node 3) | Continuous additive injection through delay lines | Per-hop resonance-window + one-shot burst routing. Source resonates 100 steps, then fires single burst through delay. No continuous pump = no blowup. Max energy now ~500. |
| 2. Reconstruction stuck at r = 0.674 | Scar-induced scattering breaks time-reversal symmetry | Damping-inverse reversal doesn't help (amplifies numerical noise). Spatial ephaptic prior helps modestly: r = 0.68 → 0.71 with conservative params, up to 0.90 at heavy partials (30%). The 0.67 ceiling is genuine wave physics with sparse scars. |
| 3. Scar saturation kills selectivity | Additive growth with beta=u² | Frogpond-correct physics: beta=\|laplacian\|, max-based growth, sparse sharp scars at Γ-shell. Plus global budget cap. Coverage drops from 0.95 → 0.06 and shape selectivity rises from ~1.0 → 3.1. |
| 4. Delay line phase rotation not isolated | Single-delay measurement was ambiguous | Multi-delay sweep with cross-correlation. δshift = δdelay confirmed exactly for delays 5-45. |

---

## Files

### `riemannnet_v2.py` — Core architecture
Five experiments that validate the fundamental mechanisms:
- **Exp 1:** Single-node discrimination. Raw ratio 5.51, normalized 3.10. Shape selectivity confirmed.
- **Exp 2:** Four-node routing with bounded energy. Correct pattern wins at most nodes.
- **Exp 3:** Time-reversal reconstruction. r = 0.67 ceiling for single-node, consistent with passive scar physics.
- **Exp 4:** Temporal orthogonality. Both patterns accessible after T_sep > N/2c.
- **Exp 5:** Delay-line phase rotation. δshift = δdelay confirmed across sweep.

### `test_ephaptic.py` — Ephaptic coupling evaluation
Tests three variants:
- Baseline (no ephaptic field)
- Scalar ephaptic (global-energy → damping modulation, Gemini's proposal)
- Spatial ephaptic (proper synergetic slaving, Pinotsis-Miller)

**Key finding:** scalar AGC breaks shape selectivity (6.11 → 0.96). Spatial ephaptic improves reconstruction modestly when needed (up to +0.26 correlation at heavy partials). Gemini's claimed +0.28 gain in the simple case was over-optimistic; the real effect is parameter-sensitive and pattern-dependent.

### `closed_loop_v2.py` — Thinking machine
The four closed-loop experiments: fixed-point convergence, competing memories (basins), dreaming (stochastic attractor exploration), active perception (memory vs ambiguous input).

This is where scars + ephaptic + templates + feedback loop produce recognizable cognitive functions.

---

## Connection to Existing Work

| Prior system | Relationship to RiemannNet |
|---|---|
| e.py (single wave pool, time reversal) | Single RiemannNet node with no routing |
| 2_image_e.py (FILO two-image) | Single node, temporal orthogonality demonstrated |
| frogpond.py (resonance engine) | Single node discrimination — correct scar physics, inherited here |
| Janus Cabbage (CVNN) | Same orthogonality principle, phase direction instead of time direction |
| Moiré Attention | Discretized single-node version of the distributed frogpond inner product |
| Clockfield-Gated Moiré Transformer | Adaptive Γ gating on the Moiré attention head — same physics, transformer context |
| Deerskin Architecture | Dendrites as Takens delay lines — Riemann lines are the physical realization |
| Pinotsis & Miller (2023) | Ephaptic slaving as spatial prior — spatial ephaptic implementation here |

---

## Honest Ledger

| Claim | Status |
|-------|--------|
| Information formula I(x) = \|∇Γ\|/Γ peaks at Γ-shell | ✓ Derived analytically, visualized in Exp 1 |
| Riemann line carries zero Clockfield information | ✓ Follows from \|∇Γ\| ≈ 0 in Region 1 |
| Delay line length rotates probe phase — δshift = δdelay | ✓ Confirmed exactly in Exp 5 |
| Temporal orthogonality condition T > N/2c | ✓ Confirmed in Exp 4 (T_sep = 100, T_orth = 65) |
| Frogpond scar physics gives shape selectivity | ✓ Normalized ratio 3.1 (was ~1.0 with amplitude scars) |
| Scar budget prevents saturation death | ✓ Four-node chain now has bounded energy |
| Spatial ephaptic field improves reconstruction from heavy partials | ✓ Up to +0.26 correlation at frac=0.3, parameter-sensitive |
| Scalar ephaptic (Gemini's proposal) helps | ✗ **Disproven.** Breaks shape discrimination. Not ephaptic coupling in any meaningful sense — it's automatic gain control. |
| Closed-loop convergence to taught pattern | ✓ r = 0.53 stable fixed point |
| Two-basin bistability in closed loop | ✓ r = ±0.57 clear attractors |
| Seed → correct basin selection | ⚠ Partial. Bistability real, seed-assignment unreliable because of wave sign-symmetry |
| Memory sustains perception through noise | ✓ r = 0.16 sustained during 11-cycle noise phase |
| Multi-node routing produces useful discrimination | ⚠ Partial. Signal propagates with bounded energy, but discrimination weakens with depth |
| This architecture outperforms standard approaches on any task | ✗ Not claimed |

---

## Problems That Remain Open

1. **Wave sign-ambiguity.** A wave equation u_tt = c²∇²u is symmetric under u → −u. Stored memories are equivalently attractors and anti-attractors. Solving this requires either Hebbian-style signed weights, complex-valued fields (CVNN/Janus), or explicit sign-breaking.

2. **Reconstruction ceiling of 0.67 at single-node, 50% partial.** This is the genuine limit of scar-mediated time reversal in a passive wave medium. Spatial ephaptic helps at heavy partials but not at moderate partials where baseline is already strong. A frequency-selective compensated reversal might push this further.

3. **Multi-node discrimination depth.** Energy is bounded and patterns propagate, but discrimination quality decays with each hop. A principled attenuation schedule (not the current coupling=1.0) could preserve more.

4. **Attractor-depth asymmetry in dreaming.** Ring attractors outcompete stripe attractors structurally. An explicit novelty term or attractor normalization would balance basin depths.

5. **Selectivity on real data.** All experiments here use synthetic shapes (square, cross, circle, diagonal, stripes, ring). MNIST-scale selectivity is untested.

---

## Quick Start

```bash
python3 riemannnet_v2.py       # 5 core experiments
python3 test_ephaptic.py        # ephaptic coupling comparison
python3 closed_loop_v2.py       # thinking machine (4 cognitive modes)
```

Plots land in `riemannnet_results/`.

Requirements: `torch`, `numpy`, `matplotlib`, `PIL`.

---

## Citation

```
Luode, A., & Claude (Anthropic). (2026). 
RiemannNet: A Wave-Physics Memory Architecture with Delay-Line Routing, 
Ephaptic Coupling, and Closed-Loop Dynamics.
GitHub: https://github.com/anttiluode/RiemannNet
```

---

## What This Computes, In One Paragraph

Given an input at time t, a memory-sculpted wave field evolves (scars constrain propagation through topological defect geometry, the ephaptic field provides a slow spatial prior, templates act as long-term attractors), produces an output at t+Δt, and that output — possibly plus external input and noise — becomes the next input. Over many cycles this is a dynamical system whose attractors are what it remembers. The output influencing the input is what makes it a thinker rather than a filter. The four experimental modes of `closed_loop_v2.py` are four regimes of one machine: recall (exp 1), discrimination (exp 2), daydreaming (exp 3), and active perception (exp 4). All from the same substrate.

---

*The Clockfield framework and all original physical insights are the work of Antti Luode.*
*Mathematical synthesis, diagnosis of the original problems, and code by Claude (Anthropic).*
*Do not hype. Do not lie. Just show.*
