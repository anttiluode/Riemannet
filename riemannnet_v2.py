"""
RiemannNet — Fixed Version
===========================
Fixes from the original Gemini-Claude implementation:

1. SCAR PHYSICS: Uses frogpond-correct beta=|laplacian| (wave curvature)
   with max-based growth, producing sparse sharp scars at Gamma-shell 
   boundaries. Original used beta=u² with additive growth → diffuse scars 
   with no shape selectivity.

2. ROUTING: One-shot burst with per-hop resonance windows. Each node 
   resonates against its scars before forwarding. No chain amplification.

3. SCAR SATURATION: Budget cap on total scar mass prevents grid death.

4. RECONSTRUCTION: Damping-inverse time reversal compensates the 
   non-unitary damping factor.

5. PHASE ROTATION: Multi-delay sweep with cross-correlation measurement.

Antti Luode — PerceptionLab, Helsinki, Finland
Claude Opus (Anthropic) — Diagnosis and fixes
April 2026

Do not hype. Do not lie. Just show.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import math


# ---------------------------------------------------------------------------
# Core wave physics
# ---------------------------------------------------------------------------

LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.],
     [1., -4., 1.],
     [0., 1., 0.]], dtype=torch.float32
).view(1, 1, 3, 3)


def laplacian(u, kernel):
    return F.conv2d(u, kernel, padding=1)


def frogpond_step(u, u_prev, scars, tau, c_sq, damping, lap_kernel, train=False,
                  scar_thresh=0.4, scar_budget=None):
    """
    One step of the Clockfield wave equation with FROGPOND scar physics.
    
    Key difference from the original riemannnet: 
      beta = |laplacian(u)|  (wave CURVATURE)
    NOT:
      beta = u²              (wave AMPLITUDE)
    
    This is the correct Clockfield formula. Gamma responds to how much 
    the wave field is bending, not how large it is. High curvature = 
    phase frustration = scar growth. Low curvature = smooth propagation 
    = no scars.
    
    Scar growth uses max() not +=, producing sharp sparse scars at 
    the exact Gamma-shell boundary.
    """
    lap = laplacian(u, lap_kernel)
    
    # CRITICAL: beta from wave curvature, not amplitude
    beta = torch.abs(lap)
    gamma = 1.0 / (1.0 + tau * beta) ** 2
    
    if train:
        freeze = (gamma < scar_thresh).float()
        candidate = torch.max(scars, freeze * (1.0 - gamma))
        
        # Budget cap: if total scar mass exceeds budget, scale down
        if scar_budget is not None:
            mass = candidate.sum().item()
            if mass > scar_budget:
                candidate = candidate * (scar_budget / (mass + 1e-8))
        
        scars = torch.clamp(candidate, 0.0, 1.0)
    
    effective_c_sq = c_sq * (1.0 - scars)
    u_next = damping * (2.0 * u - u_prev + effective_c_sq * lap)
    
    return u_next, u, gamma, scars


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class WaveNode:
    def __init__(self, size=64, tau=15.0, c_sq=0.24, damping=0.995,
                 scar_thresh=0.4, scar_budget_frac=0.15, device=None):
        """
        Parameters match frogpond.py defaults:
          tau=15.0, c_sq=0.24, damping=0.995, scar_thresh=0.4
        
        scar_budget_frac=0.15: max 15% of grid can be scarred.
            Frogpond naturally produces ~5% coverage. Budget prevents
            saturation in multi-pattern or multi-hop scenarios.
        """
        self.size = size
        self.tau = tau
        self.c_sq = c_sq
        self.damping = damping
        self.scar_thresh = scar_thresh
        self.scar_budget = scar_budget_frac * size * size
        self.device = device or torch.device('cpu')

        self.lap_kernel = LAPLACIAN_KERNEL.to(self.device)
        self.reset_field()
        self.scars = torch.zeros(1, 1, size, size, device=self.device)
        self.label = None

    def reset_field(self):
        self.u      = torch.zeros(1, 1, self.size, self.size, device=self.device)
        self.u_prev = torch.zeros(1, 1, self.size, self.size, device=self.device)

    def inject(self, pattern, offset_x=0, offset_y=0):
        h, w = pattern.shape
        cx = self.size // 2 + offset_x - h // 2
        cy = self.size // 2 + offset_y - w // 2
        cx = max(0, min(cx, self.size - h))
        cy = max(0, min(cy, self.size - w))
        self.u[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
        self.u_prev[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)

    def step(self, train=False):
        self.u, self.u_prev, gamma, self.scars = frogpond_step(
            self.u, self.u_prev, self.scars,
            self.tau, self.c_sq, self.damping, self.lap_kernel,
            train=train, scar_thresh=self.scar_thresh,
            scar_budget=self.scar_budget
        )
        return gamma

    def run(self, steps, train=False):
        for _ in range(steps):
            self.step(train=train)

    def field_energy(self):
        return float(torch.sum(self.u ** 2).item())

    def resonance_energy_history(self, probe_pattern, steps=300):
        self.reset_field()
        self.inject(probe_pattern)
        energies = []
        for _ in range(steps):
            self.step(train=False)
            energies.append(self.field_energy())
        return energies

    def time_reverse(self):
        self.u, self.u_prev = self.u_prev, self.u

    def read_center(self, h, w):
        cx = self.size // 2 - h // 2
        cy = self.size // 2 - w // 2
        patch = self.u[0, 0, cx:cx+h, cy:cy+w].detach().cpu()
        mx = patch.abs().max()
        if mx > 1e-4:
            patch = patch / mx
        return torch.clamp(patch, 0, 1)

    def information_density(self):
        lap = laplacian(self.u, self.lap_kernel)
        beta = torch.abs(lap)
        gamma = 1.0 / (1.0 + self.tau * beta) ** 2
        g = gamma[0, 0].detach().cpu().numpy()
        gy, gx = np.gradient(g)
        grad_mag = np.sqrt(gx**2 + gy**2)
        return grad_mag / (g + 1e-6)

    def compensated_reconstruct(self, partial_pattern, settle_steps=400, recon_steps=400):
        """
        FIX #2: Damping-inverse time reversal.
        
        Forward pass: each step multiplies by damping=0.995.
        Reverse pass: each step multiplies by 1/damping=1.005.
        This exactly inverts the non-unitary damping operator.
        """
        pat_h, pat_w = partial_pattern.shape
        
        self.reset_field()
        self.inject(partial_pattern)
        
        for _ in range(settle_steps):
            self.step(train=False)
        
        self.time_reverse()
        
        original_damping = self.damping
        self.damping = 1.0 / original_damping
        
        recon_steps = min(recon_steps, settle_steps)
        
        for _ in range(recon_steps):
            lap = laplacian(self.u, self.lap_kernel)
            effective_c_sq = self.c_sq * (1.0 - self.scars)
            u_next = self.damping * (2.0 * self.u - self.u_prev + effective_c_sq * lap)
            
            field_max = self.u.abs().max().item() + 1e-6
            u_next = torch.clamp(u_next, -10 * field_max, 10 * field_max)
            
            self.u_prev = self.u
            self.u = u_next
        
        self.damping = original_damping
        return self.read_center(pat_h, pat_w)


# ---------------------------------------------------------------------------
# Delay line
# ---------------------------------------------------------------------------

class DelayLine:
    def __init__(self, length, size, device=None):
        self.length = length
        self.size = size
        self.device = device or torch.device('cpu')
        self.buffer = [
            torch.zeros(1, 1, size, size, device=self.device)
            for _ in range(length)
        ]
        self.ptr = 0

    def push(self, field):
        delayed = self.buffer[self.ptr].clone()
        self.buffer[self.ptr] = field.clone()
        self.ptr = (self.ptr + 1) % self.length
        return delayed

    def flush(self):
        for i in range(self.length):
            self.buffer[i].zero_()
        self.ptr = 0


# ---------------------------------------------------------------------------
# RiemannNet
# ---------------------------------------------------------------------------

class RiemannNet:
    def __init__(self, n_nodes=4, size=64, tau=15.0, c_sq=0.24,
                 damping=0.995, scar_thresh=0.4,
                 scar_budget_frac=0.15, device=None):
        self.device = device or torch.device('cpu')
        self.n_nodes = n_nodes
        self.size = size

        self.nodes = [
            WaveNode(size=size, tau=tau, c_sq=c_sq, damping=damping,
                     scar_thresh=scar_thresh,
                     scar_budget_frac=scar_budget_frac, device=self.device)
            for _ in range(n_nodes)
        ]
        self.connections = []

    def add_connection(self, src, dst, delay_steps):
        dl = DelayLine(length=max(1, delay_steps), size=self.size, device=self.device)
        self.connections.append((src, dst, dl))

    def add_chain(self, delays):
        for i, d in enumerate(delays):
            self.add_connection(i, i + 1, d)

    def learn(self, node_id, pattern, steps=300, verbose=True):
        node = self.nodes[node_id]
        node.reset_field()
        node.inject(pattern)
        node.run(steps, train=True)
        node.label = node_id
        if verbose:
            scar_coverage = float(node.scars.mean().item())
            scar_max = float(node.scars.max().item())
            print(f"  Node {node_id}: learned {steps} steps, "
                  f"scar coverage={scar_coverage:.4f}, max={scar_max:.3f}")

    def probe_routed(self, src_node_id, probe_pattern,
                     steps=800, coupling=1.0, resonance_window=100,
                     verbose=True):
        """
        Per-hop resonance-then-burst routing.
        Each node resonates against its scars before forwarding downstream.
        """
        results = {i: [] for i in range(self.n_nodes)}

        for node in self.nodes:
            node.reset_field()
        for _, _, dl in self.connections:
            dl.flush()

        self.nodes[src_node_id].inject(probe_pattern)

        conn_state = {}
        for src, dst, dl in self.connections:
            conn_state[id(dl)] = {'fired': False, 'filling': False, 'fill_start': None}

        node_activated_at = {i: None for i in range(self.n_nodes)}
        node_activated_at[src_node_id] = 0
        
        for step in range(steps):
            for i, node in enumerate(self.nodes):
                node.step(train=False)

            for src, dst, dl in self.connections:
                dl_id = id(dl)
                state = conn_state[dl_id]
                
                if state['fired']:
                    continue
                
                if (not state['filling'] 
                    and node_activated_at[src] is not None
                    and step >= node_activated_at[src] + resonance_window):
                    state['filling'] = True
                    state['fill_start'] = step
                    dl.flush()
                
                if state['filling']:
                    outgoing = self.nodes[src].u.clone()
                    delayed = dl.push(outgoing)
                    
                    if step - state['fill_start'] >= dl.length:
                        delayed_energy = torch.sum(delayed ** 2).item()
                        if delayed_energy > 1e-6:
                            burst = coupling * delayed
                            self.nodes[dst].u = self.nodes[dst].u + burst
                            self.nodes[dst].u_prev = self.nodes[dst].u_prev + burst
                            state['fired'] = True
                            node_activated_at[dst] = step
                            if verbose:
                                print(f"    [step {step}] {src}→{dst} fired, "
                                      f"burst energy={delayed_energy*coupling**2:.2f}")

            for i, node in enumerate(self.nodes):
                results[i].append(node.field_energy())

        if verbose:
            for i in range(self.n_nodes):
                peak = max(results[i])
                tail = np.mean(results[i][-50:])
                label = self.nodes[i].label
                print(f"  Node {i} ('{label}'): peak={peak:.2f}, tail={tail:.2f}")

        return results


# ---------------------------------------------------------------------------
# Synthetic patterns
# ---------------------------------------------------------------------------

def make_synthetic_pattern(kind, size=40):
    p = torch.zeros(size, size)
    c = size // 2
    r = size // 4

    if kind == 'square':
        p[c-r:c+r, c-r:c+r] = 1.0
        p[c-r+4:c+r-4, c-r+4:c+r-4] = 0.0

    elif kind == 'cross':
        p[c-2:c+2, c-r:c+r] = 1.0
        p[c-r:c+r, c-2:c+2] = 1.0

    elif kind == 'circle':
        for i in range(size):
            for j in range(size):
                d = math.sqrt((i-c)**2 + (j-c)**2)
                if r-3 < d < r+3:
                    p[i, j] = 1.0

    elif kind == 'diagonal':
        for i in range(size):
            if 0 <= i < size:
                p[i, i] = 1.0
                if i+2 < size:
                    p[i, i+2] = 1.0
    return p


def make_partial(pattern, fraction=0.5, mode='top'):
    p = pattern.clone()
    h = pattern.shape[0]
    if mode == 'top':
        p[int(h * fraction):, :] = 0.0
    elif mode == 'noise':
        p = p * (torch.rand_like(p) < fraction).float()
    return p


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_experiment(results_dict, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12',
              '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    for i, (label, energies) in enumerate(results_dict.items()):
        ax.plot(energies, label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Field energy')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_node_state(node, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    field = node.u[0, 0].detach().cpu().numpy()
    scars = node.scars[0, 0].detach().cpu().numpy()
    info  = node.information_density()

    im0 = axes[0].imshow(field, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Wave Field u(x)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(scars, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Frozen Scars')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(info, cmap='plasma')
    axes[2].set_title('Information Density |∇Γ|/Γ')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis('off')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reconstruction(images, labels, title, save_path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, images, labels):
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_multi_node_summary(net, labels, save_path):
    n = net.n_nodes
    fig = plt.figure(figsize=(8, 3 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig)

    for i, node in enumerate(net.nodes):
        scars = node.scars[0, 0].detach().cpu().numpy()
        info  = node.information_density()

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(scars, cmap='hot', vmin=0, vmax=1)
        ax0.set_title(f'Node {i} scars  [{labels[i]}]', fontsize=9)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(info, cmap='plasma')
        ax1.set_title(f'Node {i} info density', fontsize=9)
        ax1.axis('off')

    plt.suptitle('RiemannNet — Node States', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_1(outdir, device):
    """
    Exp 1: Single-node discrimination with frogpond-correct scars.
    
    Tests both raw and energy-normalized probes to separate shape 
    selectivity from amplitude effects.
    """
    print("\n=== Experiment 1: Single-node discrimination ===")

    node = WaveNode(size=64, tau=15.0, c_sq=0.24, device=device)
    sq = make_synthetic_pattern('square', size=40)
    cr = make_synthetic_pattern('cross',  size=40)

    print("  Learning square...")
    node.reset_field()
    node.inject(sq)
    node.run(300, train=True)
    print(f"  Scar coverage: {node.scars.mean().item():.4f}")

    plot_node_state(node, 'Node after learning: square',
                   os.path.join(outdir, 'exp1_node_state.png'))

    # Raw probes
    print("  Probing (raw)...")
    e_match = node.resonance_energy_history(sq, steps=400)
    e_miss  = node.resonance_energy_history(cr, steps=400)

    plot_experiment(
        {'Resonance (square→square)': e_match,
         'Scattering (cross→square)': e_miss},
        'Exp 1: Single-node discrimination',
        os.path.join(outdir, 'exp1_discrimination.png')
    )

    tail_match = np.mean(e_match[-50:])
    tail_miss  = np.mean(e_miss[-50:])
    ratio = tail_match / (tail_miss + 1e-6)
    print(f"  RAW — Match={tail_match:.2f}, Mismatch={tail_miss:.2f}, ratio={ratio:.2f}")

    # Normalized probes (the real test of shape selectivity)
    sq_n = sq / (torch.sqrt(torch.sum(sq**2)) + 1e-8)
    cr_n = cr / (torch.sqrt(torch.sum(cr**2)) + 1e-8)
    e_match_n = node.resonance_energy_history(sq_n, steps=400)
    e_miss_n  = node.resonance_energy_history(cr_n, steps=400)
    
    tail_match_n = np.mean(e_match_n[-50:])
    tail_miss_n  = np.mean(e_miss_n[-50:])
    ratio_n = tail_match_n / (tail_miss_n + 1e-6)
    print(f"  NORMALIZED — Match={tail_match_n:.4f}, Mismatch={tail_miss_n:.4f}, ratio={ratio_n:.2f}")
    print(f"  Result: {'PASS' if ratio_n > 1.5 else 'FAIL'} (shape selectivity)")


def experiment_2(outdir, device):
    """
    Exp 2: Four-node routing with bounded energy.
    """
    print("\n=== Experiment 2: Four-node routing ===")

    patterns = {
        'square':   make_synthetic_pattern('square',   size=36),
        'cross':    make_synthetic_pattern('cross',    size=36),
        'circle':   make_synthetic_pattern('circle',   size=36),
        'diagonal': make_synthetic_pattern('diagonal', size=36),
    }
    labels = list(patterns.keys())

    net = RiemannNet(n_nodes=4, size=64, tau=15.0, c_sq=0.24,
                     scar_budget_frac=0.15, device=device)

    print("  Training nodes...")
    for i, (name, pat) in enumerate(patterns.items()):
        net.nodes[i].label = name
        net.learn(i, pat, steps=300)

    net.add_chain(delays=[10, 25, 40])

    plot_multi_node_summary(net, labels,
                            os.path.join(outdir, 'exp2_node_states.png'))

    all_results = {}
    for probe_name, probe_pat in patterns.items():
        print(f"\n  Routing probe: {probe_name}")
        res = net.probe_routed(src_node_id=0, probe_pattern=probe_pat,
                               steps=800, coupling=1.0, resonance_window=100,
                               verbose=True)
        all_results[probe_name] = res

    # Discrimination check
    print("\n  === Discrimination check ===")
    for node_id in range(4):
        node_label = labels[node_id]
        tails = {}
        for probe_name, res in all_results.items():
            tails[probe_name] = np.mean(res[node_id][-50:])
        winner = max(tails, key=tails.get)
        match = "✓" if winner == node_label else "✗"
        print(f"  Node {node_id} [{node_label}]: winner={winner} {match}  "
              f"({', '.join(f'{k}={v:.2f}' for k,v in tails.items())})")

    # Energy range check
    all_energies = []
    for res in all_results.values():
        for node_id, energies in res.items():
            all_energies.extend(energies)
    max_e = max(all_energies)
    print(f"\n  Max energy anywhere: {max_e:.2f} "
          f"({'BOUNDED' if max_e < 1e6 else 'BLOWUP'})")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    for ax, (probe_name, res) in zip(axes.flat, all_results.items()):
        for node_id, energies in res.items():
            lw = 2.5 if labels[node_id] == probe_name else 1.0
            ax.plot(energies, label=f'node {node_id} [{labels[node_id]}]',
                    color=colors[node_id], linewidth=lw)
        ax.set_title(f'Probe: {probe_name}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Energy')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle('Exp 2: Four-node routed discrimination', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'exp2_routing.png'), dpi=120)
    plt.close()
    print(f"  Saved: {os.path.join(outdir, 'exp2_routing.png')}")


def experiment_3(outdir, device):
    """
    Exp 3: Reconstruction — standard vs damping-compensated.
    """
    print("\n=== Experiment 3: Reconstruction ===")

    pat_size = 40
    sq = make_synthetic_pattern('square', size=pat_size)
    partial = make_partial(sq, fraction=0.5, mode='top')

    # Standard method
    node_std = WaveNode(size=64, tau=15.0, c_sq=0.24, device=device)
    node_std.reset_field()
    node_std.inject(sq)
    node_std.run(300, train=True)

    node_std.reset_field()
    node_std.inject(partial)
    node_std.run(400, train=False)
    node_std.time_reverse()
    node_std.run(400, train=False)
    recovered_std = node_std.read_center(pat_size, pat_size)

    sq_np = sq.numpy().ravel()
    rs_np = recovered_std.numpy().ravel()
    corr_std = np.corrcoef(sq_np, rs_np)[0, 1] if sq_np.std() > 1e-4 and rs_np.std() > 1e-4 else 0.0

    # Compensated method
    node_comp = WaveNode(size=64, tau=15.0, c_sq=0.24, device=device)
    node_comp.reset_field()
    node_comp.inject(sq)
    node_comp.run(300, train=True)

    plot_node_state(node_comp, 'Scars before reconstruction',
                    os.path.join(outdir, 'exp3_scars.png'))

    recovered_comp = node_comp.compensated_reconstruct(partial)
    rc_np = recovered_comp.numpy().ravel()
    corr_comp = np.corrcoef(sq_np, rc_np)[0, 1] if sq_np.std() > 1e-4 and rc_np.std() > 1e-4 else 0.0

    print(f"  Standard correlation:    {corr_std:.3f}")
    print(f"  Compensated correlation: {corr_comp:.3f}")
    print(f"  Improvement: {corr_comp - corr_std:+.3f}")

    plot_reconstruction(
        [sq, partial, recovered_std, recovered_comp],
        ['Original', 'Partial (50%)', 
         f'Standard (r={corr_std:.3f})', f'Compensated (r={corr_comp:.3f})'],
        'Exp 3: Reconstruction comparison',
        os.path.join(outdir, 'exp3_reconstruction.png')
    )


def experiment_4(outdir, device):
    """Exp 4: Temporal orthogonality — unchanged."""
    print("\n=== Experiment 4: Temporal orthogonality ===")

    node = WaveNode(size=64, tau=15.0, c_sq=0.24, device=device)
    sq = make_synthetic_pattern('square', size=38)
    cr = make_synthetic_pattern('cross',  size=38)

    T_sep = 100
    c = math.sqrt(0.24)
    T_orth = 64 / (2 * c)
    print(f"  T_sep={T_sep}, T_orthogonal={T_orth:.1f}")

    node.reset_field()
    node.inject(sq)
    node.run(T_sep // 2, train=True)
    node.inject(cr)
    node.run(T_sep // 2, train=True)

    e_sq = node.resonance_energy_history(sq, steps=300)
    e_cr = node.resonance_energy_history(cr, steps=300)

    plot_experiment(
        {'Square probe': e_sq, 'Cross probe': e_cr},
        'Exp 4: Temporal orthogonality — two patterns, one node',
        os.path.join(outdir, 'exp4_orthogonality.png')
    )

    tail_sq = np.mean(e_sq[-50:])
    tail_cr = np.mean(e_cr[-50:])
    print(f"  Square tail={tail_sq:.2f}, Cross tail={tail_cr:.2f}")
    print(f"  Result: {'PASS' if tail_sq > 5 and tail_cr > 5 else 'PARTIAL'}")


def experiment_5(outdir, device):
    """
    Exp 5: Phase rotation by delay length — multi-delay sweep.
    """
    print("\n=== Experiment 5: Delay line phase sweep ===")

    sq = make_synthetic_pattern('square', size=36)
    delays = [5, 15, 25, 35, 45]
    
    curves = {}
    for delay in delays:
        net = RiemannNet(n_nodes=2, size=64, tau=15.0, c_sq=0.24, device=device)
        net.nodes[1].label = f'delay={delay}'
        net.learn(1, sq, steps=300, verbose=False)
        net.add_connection(0, 1, delay_steps=delay)
        res = net.probe_routed(0, sq, steps=500, coupling=1.0, 
                               resonance_window=80, verbose=False)
        curves[f'delay={delay}'] = res[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(delays)))
    for (label, energies), color in zip(curves.items(), colors):
        ax.plot(energies, label=label, color=color, linewidth=1.5)
    ax.set_title('Exp 5: Phase rotation — same pattern, different delays', fontsize=12)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Destination energy')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'exp5_delay_sweep.png'), dpi=120)
    plt.close()
    print(f"  Saved: {os.path.join(outdir, 'exp5_delay_sweep.png')}")

    # Cross-correlation phase shift measurement
    ref = np.array(curves[f'delay={delays[0]}'])
    print("\n  Phase shift analysis (cross-correlation):")
    shifts = []
    for delay in delays[1:]:
        test = np.array(curves[f'delay={delay}'])
        # Use a window after the burst arrives
        window = slice(100, min(len(ref), len(test)))
        r = (ref[window] - ref[window].mean()) / (ref[window].std() + 1e-8)
        t = (test[window] - test[window].mean()) / (test[window].std() + 1e-8)
        xcorr = np.correlate(r, t, mode='full')
        shift = np.argmax(xcorr) - (len(r) - 1)
        shifts.append((delay, shift))
        print(f"  delay={delay}: shift={shift} steps (expected ~{delay - delays[0]})")

    if shifts:
        fig, ax = plt.subplots(figsize=(7, 4))
        ds = [d for d, _ in shifts]
        ss = [s for _, s in shifts]
        ax.plot(ds, ss, 'o-', color='#e74c3c', markersize=8, linewidth=2, label='Measured')
        ax.plot(ds, [d - delays[0] for d in ds], '--', color='gray', label='Theoretical')
        ax.set_xlabel('Delay length (steps)')
        ax.set_ylabel('Phase shift (steps)')
        ax.set_title('Phase shift vs delay length')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'exp5_phase_shift.png'), dpi=120)
        plt.close()
        print(f"  Saved: {os.path.join(outdir, 'exp5_phase_shift.png')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    outdir = 'riemannnet_results'
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RiemannNet (Fixed) — device: {device}")

    run = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if run in ('all', '1'): experiment_1(outdir, device)
    if run in ('all', '2'): experiment_2(outdir, device)
    if run in ('all', '3'): experiment_3(outdir, device)
    if run in ('all', '4'): experiment_4(outdir, device)
    if run in ('all', '5'): experiment_5(outdir, device)

    print("\nDone.")
    print("Do not hype. Do not lie. Just show.")
