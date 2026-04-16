"""
Closed-Loop RiemannNet v2 — Whole-Field Architecture
====================================================
The system thinks by:
  1. Field holds a state u(x, t)
  2. Measure the field → produce an output pattern
  3. Output pattern is injected back as a probe (with gain, delay, noise)
  4. The wave dynamics (with scars + ephaptic) evolve
  5. Repeat

No separate regions. The entire grid is the 'brain'. The 'output' is 
a readout of the current field state. The 'input' is a re-injection 
of that readout after a delay.

This matches how the frogpond actually works: probe → ringing → 
read total field. Closing the loop means: ringing → new probe.

What is computed:
  - Fixed points: patterns where readout(t) ≈ readout(t+delay)
  - Limit cycles: patterns that periodically return
  - Strange attractors: patterns that drift chaotically through memory
  - With noise: sampling over the attractor manifold
  
The scars store WHICH patterns are stable (they are the topological 
skeleton the wave settles into). The ephaptic field is the slow 
low-pass memory of recent activity — it guides which attractor 
the system currently sits in.

Do not hype. Do not lie. Just show.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import math

from riemannnet_v2 import LAPLACIAN_KERNEL, laplacian, make_synthetic_pattern


class ThinkingNode:
    """
    A wave node that runs in a closed loop — its output at time t 
    becomes its input at time t+1 (with a delay, optional noise).
    
    Think of it as: the brain is running. Every 'moment of thought' 
    is one cycle: the current field is measured, reshaped slightly,
    re-injected. The scars set the landscape; the ephaptic field 
    remembers where we recently were.
    """
    
    def __init__(self, size=64, tau=15.0, c_sq=0.24, damping=0.995,
                 scar_thresh=0.4, scar_budget_frac=0.15,
                 eph_tau=0.1, eph_blur=3.0, eph_pull=0.01,
                 feedback_gain=0.7, feedback_noise=0.0,
                 device=None):
        self.size = size
        self.tau = tau
        self.c_sq = c_sq
        self.damping = damping
        self.scar_thresh = scar_thresh
        self.scar_budget = scar_budget_frac * size * size
        self.device = device or torch.device('cpu')
        
        self.eph_tau = eph_tau
        self.eph_blur = eph_blur
        self.eph_pull = eph_pull
        
        self.feedback_gain = feedback_gain
        self.feedback_noise = feedback_noise
        
        self.lap_kernel = LAPLACIAN_KERNEL.to(self.device)
        
        # Gaussian blur kernel
        ks = int(2 * math.ceil(3 * eph_blur) + 1)
        coords = torch.arange(ks, dtype=torch.float32) - ks // 2
        g = torch.exp(-coords**2 / (2 * eph_blur**2))
        g = g / g.sum()
        self.blur_k = g.view(1, 1, -1).to(self.device)
        self.blur_ks = ks
        
        self.u      = torch.zeros(1, 1, size, size, device=self.device)
        self.u_prev = torch.zeros(1, 1, size, size, device=self.device)
        self.scars  = torch.zeros(1, 1, size, size, device=self.device)
        self.ephaptic = torch.zeros(1, 1, size, size, device=self.device)
        
        # Template memories: explicit storage of learned patterns.
        # Biologically, this represents long-term potentiation — the stable 
        # weight configuration that persists over days. In the Clockfield 
        # framework, these are the topological modes locked in by the Γ→0 
        # singularities. Algorithmically, the scars tell you WHERE boundaries 
        # are; the templates tell you WHICH pattern of signs is stored there.
        self.templates = []  # list of (size, size) tensors
        self.template_strength = 0.0  # how much template biases the dynamics
        self.template_temp = 3.0       # softmax temperature (higher = sharper)
    
    def reset_waves(self):
        self.u.zero_()
        self.u_prev.zero_()
    
    def reset_short_term(self):
        self.reset_waves()
        self.ephaptic.zero_()
    
    def reset_all(self):
        self.reset_short_term()
        self.scars.zero_()
    
    def _blur(self, f):
        p = self.blur_ks // 2
        x = F.conv2d(f, self.blur_k.view(1, 1, 1, -1), padding=(0, p))
        x = F.conv2d(x, self.blur_k.view(1, 1, -1, 1), padding=(p, 0))
        return x
    
    def inject_center(self, pattern):
        """Center-inject a pattern."""
        h, w = pattern.shape
        cx = self.size // 2 - h // 2
        cy = self.size // 2 - w // 2
        self.u[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
        self.u_prev[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
    
    def inject_full(self, pattern, additive=True):
        """Inject full-grid pattern."""
        p = pattern.to(self.device).view(1, 1, self.size, self.size)
        if additive:
            self.u = self.u + p
            self.u_prev = self.u_prev + p
        else:
            self.u = p.clone()
            self.u_prev = p.clone()
    
    def store_template(self, pattern):
        """
        Store a pattern as a long-term template memory.
        During closed-loop operation, templates act as weak biases that 
        make themselves attractors. The system chooses which template to 
        sit in based on which one best matches the current field.
        """
        p = pattern.to(self.device).view(1, 1, self.size, self.size).clone()
        self.templates.append(p)
    
    def _template_mix(self, temperature=5.0):
        """
        Softmax-weighted mix of all templates based on correlation with 
        current field. Returns weighted template (or None if no templates).
        
        temperature: higher = sharper (more winner-take-all), 
                     lower = softer (more mixing).
        Also returns the per-template weights so callers can inspect.
        """
        if not self.templates:
            return None, None
        
        u_flat = self.u.view(-1)
        u_centered = u_flat - u_flat.mean()
        u_norm = u_centered / (u_centered.norm() + 1e-8)
        
        corrs = []
        for t in self.templates:
            t_flat = t.view(-1)
            t_centered = t_flat - t_flat.mean()
            t_norm = t_centered / (t_centered.norm() + 1e-8)
            corrs.append((u_norm * t_norm).sum().item())
        
        # Softmax weighting
        corrs_t = torch.tensor(corrs)
        weights = torch.softmax(temperature * corrs_t, dim=0)
        
        mixed = torch.zeros_like(self.templates[0])
        for w, t in zip(weights, self.templates):
            mixed = mixed + w * t
        
        return mixed, weights.numpy()
    
    def _best_template_match(self):
        """Return the template with highest correlation to current field."""
        if not self.templates:
            return None
        u_flat = self.u.view(-1)
        u_mean = u_flat.mean()
        u_centered = u_flat - u_mean
        u_norm = u_centered / (u_centered.norm() + 1e-8)
        
        best_corr = -2.0
        best_template = None
        for t in self.templates:
            t_flat = t.view(-1)
            t_centered = t_flat - t_flat.mean()
            t_norm = t_centered / (t_centered.norm() + 1e-8)
            corr = (u_norm * t_norm).sum().item()
            if corr > best_corr:
                best_corr = corr
                best_template = t
        return best_template, best_corr
    
    def step(self, train=False, ephaptic_on=True):
        lap = laplacian(self.u, self.lap_kernel)
        beta = torch.abs(lap)
        gamma = 1.0 / (1.0 + self.tau * beta) ** 2
        
        if train:
            freeze = (gamma < self.scar_thresh).float()
            cand = torch.max(self.scars, freeze * (1.0 - gamma))
            mass = cand.sum().item()
            if mass > self.scar_budget:
                cand = cand * (self.scar_budget / (mass + 1e-8))
            self.scars = torch.clamp(cand, 0.0, 1.0)
        
        if ephaptic_on:
            blurred = self._blur(self.u)
            self.ephaptic = ((1 - self.eph_tau) * self.ephaptic 
                             + self.eph_tau * blurred)
        
        eff_c_sq = self.c_sq * (1.0 - self.scars)
        u_next = self.damping * (2.0 * self.u - self.u_prev + eff_c_sq * lap)
        
        if ephaptic_on:
            u_next = u_next + self.eph_pull * (self.ephaptic - u_next)
        
        # Template bias: softmax-weighted mixture of all templates,
        # weighted by how well each matches the current field.
        # This creates soft attractor dynamics — leaning toward a template
        # when there's clear evidence, staying balanced when ambiguous.
        if self.templates and self.template_strength > 0:
            mix = self._template_mix(temperature=self.template_temp)
            if mix is not None and mix[0] is not None:
                mixed_tmpl, _ = mix
                u_next = u_next + self.template_strength * (mixed_tmpl - u_next)
        
        self.u_prev = self.u
        self.u = u_next
        return gamma
    
    def run(self, steps, train=False, ephaptic_on=True):
        for _ in range(steps):
            self.step(train=train, ephaptic_on=ephaptic_on)
    
    def field_energy(self):
        return float(torch.sum(self.u ** 2).item())
    
    def get_field(self):
        return self.u[0, 0].detach().cpu().clone()
    
    # -----------------------------------------------------------
    # THE THINKING LOOP
    # -----------------------------------------------------------
    
    def think_step(self, external_input=None, steps_per_cycle=50,
                   ephaptic_on=True, target_energy=50.0):
        """
        One cycle of thinking:
          1. Run the wave forward for steps_per_cycle
          2. Read out the current field state
          3. Re-inject a normalized version as new probe
          4. Add external input if provided
          5. Return the readout
        
        Stability: the field is energy-normalized to `target_energy` each 
        cycle. This prevents unbounded growth in closed-loop mode and also 
        prevents the field from dying out. The scars + ephaptic determine 
        the SHAPE; the normalization keeps AMPLITUDE bounded.
        """
        # Forward evolution
        for _ in range(steps_per_cycle):
            self.step(train=False, ephaptic_on=ephaptic_on)
        
        # Readout: current field
        readout = self.get_field()
        
        # Energy normalization: rescale the current field to target_energy
        current_energy = self.field_energy()
        if current_energy > 1e-8:
            scale = math.sqrt(target_energy / current_energy)
            self.u = self.u * scale
            self.u_prev = self.u_prev * scale
        
        # Apply feedback gain (amplify/attenuate relative to target)
        self.u = self.feedback_gain * self.u
        self.u_prev = self.feedback_gain * self.u_prev
        
        # Add noise if specified (into the field, not the readout)
        if self.feedback_noise > 0:
            noise = self.feedback_noise * torch.randn_like(self.u)
            self.u = self.u + noise
            self.u_prev = self.u_prev + noise
        
        # Add external input if provided
        if external_input is not None:
            ext = external_input.view(1, 1, self.size, self.size).to(self.device)
            self.u = self.u + ext
            self.u_prev = self.u_prev + ext
        
        return readout


# ---------------------------------------------------------------------------
# Full-grid patterns
# ---------------------------------------------------------------------------

def make_full_pattern(kind, size=64):
    """Make a full-grid pattern."""
    p = torch.zeros(size, size)
    c = size // 2
    
    if kind == 'horiz_stripes':
        for i in range(size):
            if (i // 6) % 2 == 0:
                p[i, :] = 1.0
    
    elif kind == 'vert_stripes':
        for j in range(size):
            if (j // 6) % 2 == 0:
                p[:, j] = 1.0
    
    elif kind == 'diag_stripes':
        for i in range(size):
            for j in range(size):
                if ((i + j) // 6) % 2 == 0:
                    p[i, j] = 1.0
    
    elif kind == 'checker':
        for i in range(size):
            for j in range(size):
                if ((i // 6) + (j // 6)) % 2 == 0:
                    p[i, j] = 1.0
    
    elif kind == 'spot_center':
        for i in range(size):
            for j in range(size):
                d2 = (i - c)**2 + (j - c)**2
                p[i, j] = math.exp(-d2 / (2 * (size/8)**2))
    
    elif kind == 'ring':
        for i in range(size):
            for j in range(size):
                d = math.sqrt((i - c)**2 + (j - c)**2)
                if size/4 - 3 < d < size/4 + 3:
                    p[i, j] = 1.0
    
    return p


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def exp1_convergence(outdir, device):
    """
    Train on one pattern, close loop, see if random init converges there.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fixed-point convergence")
    print("=" * 70)
    
    node = ThinkingNode(size=64, feedback_gain=0.85, feedback_noise=0.0,
                         scar_budget_frac=0.15, device=device)
    
    # Teach: inject pattern, let scars grow
    pattern = make_full_pattern('horiz_stripes', size=64)
    print("  Teaching: horizontal stripes")
    node.reset_waves()
    node.inject_full(pattern, additive=False)
    node.run(400, train=True, ephaptic_on=True)
    print(f"  Scar coverage: {node.scars.mean().item():.4f}")
    
    # Store pattern as long-term template (the "engram")
    node.store_template(pattern)
    node.template_strength = 0.008  # moderate bias
    node.template_temp = 5.0         # sharp (only one template anyway)
    
    # Close loop from random init
    print("\n  Closing loop from RANDOM init...")
    node.reset_short_term()  # keep scars
    torch.manual_seed(7)
    node.inject_full(0.3 * torch.randn(64, 64), additive=False)
    
    trajectory = []
    corrs = []
    energies = []
    
    for cycle in range(40):
        out = node.think_step(steps_per_cycle=40, ephaptic_on=True)
        trajectory.append(out.clone())
        
        c = np.corrcoef(out.numpy().ravel(), pattern.numpy().ravel())[0, 1]
        corrs.append(c if not np.isnan(c) else 0.0)
        energies.append(node.field_energy())
    
    print(f"  Final correlation with taught pattern: {corrs[-1]:.3f}")
    print(f"  Correlation trajectory (every 5 cycles): " + 
          ", ".join(f"{corrs[i]:+.2f}" for i in range(0, 40, 5)))
    print(f"  Final energy: {energies[-1]:.2f}")
    
    # Plot
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 8, figure=fig)
    
    show = [0, 3, 6, 10, 15, 20, 30, 39]
    for i, c in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        m = trajectory[c].abs().max().item() + 1e-6
        ax.imshow(trajectory[c].numpy() / m, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'cycle {c}\nr={corrs[c]:+.2f}', fontsize=9)
        ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(pattern.numpy(), cmap='gray')
    ax.set_title('Taught pattern'); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(node.scars[0, 0].cpu().numpy(), cmap='hot')
    ax.set_title('Scars'); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    eph = node.ephaptic[0, 0].cpu().numpy()
    em = max(abs(eph.min()), abs(eph.max())) + 1e-6
    ax.imshow(eph / em, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Ephaptic'); ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(corrs, color='#e74c3c', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Correlation')
    ax.set_title('Convergence to pattern')
    ax.grid(alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 4])
    ax.semilogy(energies, color='#2ecc71', linewidth=2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Energy (log)')
    ax.set_title('Energy (bounded?)')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Fixed-point convergence: random init → learned pattern?',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, 'think_1_convergence.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def exp2_competing_memories(outdir, device):
    """
    Train on two patterns. Test which attractor each seed falls into.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Competing memories — basins of attraction")
    print("=" * 70)
    
    node = ThinkingNode(size=64, feedback_gain=0.85, feedback_noise=0.0,
                         scar_budget_frac=0.20, device=device)
    
    horiz = make_full_pattern('horiz_stripes', size=64)
    vert  = make_full_pattern('vert_stripes', size=64)
    
    # Interleaved teaching
    print("  Teaching both patterns (interleaved)...")
    for _ in range(3):
        node.reset_waves()
        node.inject_full(horiz, additive=False)
        node.run(200, train=True, ephaptic_on=True)
        node.reset_waves()
        node.inject_full(vert, additive=False)
        node.run(200, train=True, ephaptic_on=True)
    print(f"  Scar coverage: {node.scars.mean().item():.4f}")
    
    # Store both patterns as templates
    node.store_template(horiz)
    node.store_template(vert)
    node.template_strength = 0.008
    node.template_temp = 8.0  # sharper — commit to one template
    
    seeds = {
        'horiz_hint': 0.5 * horiz + 0.3 * torch.randn(64, 64),
        'vert_hint':  0.5 * vert  + 0.3 * torch.randn(64, 64),
        'neutral':    0.4 * torch.randn(64, 64),
    }
    
    results = {}
    for name, seed in seeds.items():
        torch.manual_seed(0)  # same noise for fairness
        node.reset_short_term()
        node.inject_full(seed, additive=False)
        
        traj = []
        corrs_h = []
        corrs_v = []
        for cycle in range(30):
            out = node.think_step(steps_per_cycle=40, ephaptic_on=True)
            traj.append(out.clone())
            ch = np.corrcoef(out.numpy().ravel(), horiz.numpy().ravel())[0, 1]
            cv = np.corrcoef(out.numpy().ravel(), vert.numpy().ravel())[0, 1]
            corrs_h.append(ch if not np.isnan(ch) else 0)
            corrs_v.append(cv if not np.isnan(cv) else 0)
        
        results[name] = (traj, corrs_h, corrs_v, seed)
        print(f"  [{name}] final: corr(H)={corrs_h[-1]:+.3f}, corr(V)={corrs_v[-1]:+.3f}")
    
    # Plot
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1, 1, 1, 1.2])
    
    for row, (name, (traj, ch, cv, seed)) in enumerate(results.items()):
        ax = fig.add_subplot(gs[row, 0])
        m = seed.abs().max().item() + 1e-6
        ax.imshow(seed.numpy() / m, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'{name}\nseed'); ax.axis('off')
        
        show = [0, 5, 15, 29]
        for col, c in enumerate(show, start=1):
            ax = fig.add_subplot(gs[row, col])
            m = traj[c].abs().max().item() + 1e-6
            ax.imshow(traj[c].numpy() / m, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'c{c}\nH={ch[c]:+.2f} V={cv[c]:+.2f}', fontsize=8)
            ax.axis('off')
        
        ax = fig.add_subplot(gs[row, 5])
        ax.plot(ch, color='#e74c3c', label='H', linewidth=2)
        ax.plot(cv, color='#2ecc71', label='V', linewidth=2)
        ax.set_xlabel('cycle')
        ax.set_ylabel('corr')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Bottom row: templates + scars
    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(horiz.numpy(), cmap='gray'); ax.set_title('Horiz template'); ax.axis('off')
    ax = fig.add_subplot(gs[3, 1])
    ax.imshow(vert.numpy(), cmap='gray'); ax.set_title('Vert template'); ax.axis('off')
    ax = fig.add_subplot(gs[3, 2])
    ax.imshow(node.scars[0, 0].cpu().numpy(), cmap='hot')
    ax.set_title('Scars (both patterns)'); ax.axis('off')
    
    plt.suptitle('Two learned patterns, three seeds — which basin?', fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, 'think_2_competing.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def exp3_dreaming(outdir, device):
    """Stochastic feedback → drifting between learned patterns."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Dreaming (noisy feedback → wanders between attractors)")
    print("=" * 70)
    
    node = ThinkingNode(size=64, feedback_gain=0.70, feedback_noise=0.25,
                         scar_budget_frac=0.25, device=device)
    
    patterns = {
        'horiz': make_full_pattern('horiz_stripes', size=64),
        'vert':  make_full_pattern('vert_stripes', size=64),
        'ring':  make_full_pattern('ring', size=64),
    }
    
    for _ in range(3):
        for name, pat in patterns.items():
            node.reset_waves()
            node.inject_full(pat, additive=False)
            node.run(150, train=True, ephaptic_on=True)
    print(f"  Scar coverage: {node.scars.mean().item():.4f}")
    
    # Store all patterns as templates
    for name, pat in patterns.items():
        node.store_template(pat)
    node.template_strength = 0.003  # weaker — noise can escape
    node.template_temp = 3.0  # softer mixing
    
    print("\n  Dreaming for 80 cycles...")
    torch.manual_seed(3)
    node.reset_short_term()
    node.inject_full(0.3 * torch.randn(64, 64), additive=False)
    
    traj = []
    corrs = {k: [] for k in patterns}
    for cycle in range(80):
        out = node.think_step(steps_per_cycle=40, ephaptic_on=True)
        traj.append(out.clone())
        for name, pat in patterns.items():
            c = np.corrcoef(out.numpy().ravel(), pat.numpy().ravel())[0, 1]
            corrs[name].append(c if not np.isnan(c) else 0.0)
    
    # Which attractor per cycle?
    dominant = []
    for cycle in range(80):
        vals = {name: corrs[name][cycle] for name in patterns}
        best = max(vals, key=vals.get)
        if vals[best] > 0.2:
            dominant.append(best)
        else:
            dominant.append('none')
    
    visits = {k: dominant.count(k) for k in list(patterns) + ['none']}
    print("  Attractor visit counts:")
    for k, v in visits.items():
        print(f"    {k}: {v}/80")
    
    # Plot
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 8, figure=fig, height_ratios=[1, 1, 1.3])
    
    show = [0, 10, 20, 30, 40, 50, 60, 75]
    for i, c in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        m = traj[c].abs().max().item() + 1e-6
        ax.imshow(traj[c].numpy() / m, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'c{c}\n{dominant[c]}', fontsize=8)
        ax.axis('off')
    
    for i, (name, pat) in enumerate(patterns.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(pat.numpy(), cmap='gray')
        ax.set_title(f'Taught: {name}'); ax.axis('off')
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(node.scars[0, 0].cpu().numpy(), cmap='hot')
    ax.set_title('Scars'); ax.axis('off')
    
    ax = fig.add_subplot(gs[2, :])
    colors_ = {'horiz': '#e74c3c', 'vert': '#2ecc71', 'ring': '#3498db'}
    for name, color in colors_.items():
        ax.plot(corrs[name], label=name, color=color, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.2, color='black', linestyle=':', alpha=0.3, label='threshold')
    ax.set_xlabel('Dream cycle')
    ax.set_ylabel('Correlation')
    ax.set_title('Dream trajectory: which attractor does the system near?')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Dreaming: stochastic feedback explores attractor manifold',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, 'think_3_dreaming.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def exp4_perception(outdir, device):
    """
    External input combined with internal feedback.
    Does memory sustain perception when input becomes ambiguous?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Active perception (ext input + internal memory)")
    print("=" * 70)
    
    node = ThinkingNode(size=64, feedback_gain=0.7, feedback_noise=0.0,
                         scar_budget_frac=0.20, device=device)
    
    pattern = make_full_pattern('horiz_stripes', size=64)
    print("  Teaching: horiz stripes")
    for _ in range(3):
        node.reset_waves()
        node.inject_full(pattern, additive=False)
        node.run(300, train=True, ephaptic_on=True)
    print(f"  Scar coverage: {node.scars.mean().item():.4f}")
    
    # Store the pattern as long-term template
    node.store_template(pattern)
    node.template_strength = 0.005
    node.template_temp = 5.0
    
    # Timeline: 
    #   c=0..6:   external = pattern
    #   c=7..17:  external = noise
    #   c=18..25: external = pattern
    print("\n  Running with external input schedule...")
    node.reset_short_term()
    node.inject_full(0.2 * torch.randn(64, 64), additive=False)
    
    traj = []
    ext_tr = []
    corrs = []
    labels = []
    
    for cycle in range(26):
        if cycle < 7:
            ext = 0.6 * pattern
            lab = 'pattern'
        elif cycle < 18:
            ext = 0.3 * torch.randn(64, 64)
            lab = 'noise'
        else:
            ext = 0.6 * pattern
            lab = 'pattern'
        
        out = node.think_step(external_input=ext, steps_per_cycle=40,
                               ephaptic_on=True)
        traj.append(out.clone())
        ext_tr.append((ext.clone(), lab))
        c = np.corrcoef(out.numpy().ravel(), pattern.numpy().ravel())[0, 1]
        corrs.append(c if not np.isnan(c) else 0)
        labels.append(lab)
    
    print(f"\n  Correlation during 'pattern' phases: "
          f"start={np.mean(corrs[:7]):.3f}, end={np.mean(corrs[18:]):.3f}")
    print(f"  Correlation during 'noise' phase (cycles 7-17): "
          f"{np.mean(corrs[7:18]):.3f}")
    
    # Plot
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(3, 8, figure=fig)
    
    show = [1, 4, 7, 10, 14, 17, 20, 25]
    for i, c in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        em = ext_tr[c][0].abs().max().item() + 1e-6
        ax.imshow(ext_tr[c][0].numpy() / em, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'c{c} ext:{labels[c]}', fontsize=8)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, i])
        m = traj[c].abs().max().item() + 1e-6
        ax.imshow(traj[c].numpy() / m, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'output r={corrs[c]:+.2f}', fontsize=8)
        ax.axis('off')
    
    ax = fig.add_subplot(gs[2, :])
    ax.plot(corrs, color='#e74c3c', linewidth=2, label='output vs pattern')
    ax.axvspan(0, 6.5, alpha=0.2, color='green', label='pattern')
    ax.axvspan(6.5, 17.5, alpha=0.2, color='gray', label='noise')
    ax.axvspan(17.5, 25, alpha=0.2, color='green')
    ax.axhline(0, color='black', alpha=0.3)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Output ↔ pattern correlation')
    ax.set_title('Does memory sustain perception through the ambiguous phase?')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Active perception: external + internal',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, 'think_4_perception.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == '__main__':
    outdir = 'riemannnet_results'
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cpu')
    
    print("ThinkingNode — closed-loop RiemannNet")
    print("=" * 70)
    
    exp1_convergence(outdir, device)
    exp2_competing_memories(outdir, device)
    exp3_dreaming(outdir, device)
    exp4_perception(outdir, device)
    
    print("\n" + "=" * 70)
    print("Done.")
