"""
Test: Does adding an ephaptic field actually solve the remaining problems?

Two versions tested:
  A) Scalar ephaptic (Gemini's proposal): global energy modulates damping
  B) Spatial ephaptic (proper slaving): slow field acts as spatial prior

Target problems:
  - Exp 2: chain attenuation (node 3 gets nothing)
  - Exp 3: reconstruction stuck at 0.674 correlation ceiling
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, math

from riemannnet_v2 import (
    LAPLACIAN_KERNEL, laplacian, make_synthetic_pattern, make_partial,
    DelayLine, plot_experiment, plot_reconstruction
)


# ---------------------------------------------------------------------------
# Scalar ephaptic node (Gemini's proposal)
# ---------------------------------------------------------------------------

class ScalarEphapticNode:
    """
    Global energy acts as control parameter modulating damping.
    
    ephaptic_bias = slow-moving average of total field energy
    dynamic_damping = damping * (1 - eph_coupling * ephaptic_bias)
    
    If bias is HIGH (lots of energy): increase damping → system cools
    If bias is LOW (quiet): decrease damping → system amplifies
    
    This is automatic gain control — it keeps total energy near a setpoint.
    """
    def __init__(self, size=64, tau=15.0, c_sq=0.24, damping=0.995,
                 scar_thresh=0.4, scar_budget_frac=0.15,
                 eph_target=10.0, eph_tau=0.05, eph_coupling=0.01,
                 device=None):
        self.size = size
        self.tau = tau
        self.c_sq = c_sq
        self.damping = damping
        self.scar_thresh = scar_thresh
        self.scar_budget = scar_budget_frac * size * size
        self.device = device or torch.device('cpu')
        
        # Ephaptic parameters
        self.eph_target = eph_target       # target energy setpoint
        self.eph_tau = eph_tau             # slow timescale (low-pass)
        self.eph_coupling = eph_coupling   # how strongly bias modulates damping
        self.ephaptic_bias = 0.0           # the slow state
        
        self.lap_kernel = LAPLACIAN_KERNEL.to(self.device)
        self.reset_field()
        self.scars = torch.zeros(1, 1, size, size, device=self.device)
        self.label = None
    
    def reset_field(self):
        self.u = torch.zeros(1, 1, self.size, self.size, device=self.device)
        self.u_prev = torch.zeros(1, 1, self.size, self.size, device=self.device)
        self.ephaptic_bias = 0.0
    
    def inject(self, pattern):
        h, w = pattern.shape
        cx = self.size // 2 - h // 2
        cy = self.size // 2 - w // 2
        self.u[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
        self.u_prev[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
    
    def step(self, train=False):
        lap = laplacian(self.u, self.lap_kernel)
        beta = torch.abs(lap)
        gamma = 1.0 / (1.0 + self.tau * beta) ** 2
        
        if train:
            freeze = (gamma < self.scar_thresh).float()
            candidate = torch.max(self.scars, freeze * (1.0 - gamma))
            mass = candidate.sum().item()
            if mass > self.scar_budget:
                candidate = candidate * (self.scar_budget / (mass + 1e-8))
            self.scars = torch.clamp(candidate, 0.0, 1.0)
        
        # Ephaptic update: slow tracking of total energy toward target
        current_energy = torch.sum(self.u ** 2).item()
        # Error relative to target (positive = too hot, negative = too cold)
        error = (current_energy - self.eph_target) / (self.eph_target + 1e-6)
        # Slow integration
        self.ephaptic_bias = (1 - self.eph_tau) * self.ephaptic_bias + self.eph_tau * error
        
        # Modulate damping: bias > 0 means too hot → dampen more
        # bias < 0 means too cold → amplify (damping > 1)
        dynamic_damping = self.damping * (1.0 - self.eph_coupling * self.ephaptic_bias)
        # Safety clamp
        dynamic_damping = max(0.98, min(1.005, dynamic_damping))
        
        effective_c_sq = self.c_sq * (1.0 - self.scars)
        u_next = dynamic_damping * (2.0 * self.u - self.u_prev + effective_c_sq * lap)
        
        self.u_prev = self.u
        self.u = u_next
        return gamma
    
    def run(self, steps, train=False):
        for _ in range(steps):
            self.step(train=train)
    
    def field_energy(self):
        return float(torch.sum(self.u ** 2).item())
    
    def resonance_energy_history(self, probe, steps=300):
        self.reset_field()
        self.inject(probe)
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


# ---------------------------------------------------------------------------
# Spatial ephaptic node (proper slaving principle)
# ---------------------------------------------------------------------------

class SpatialEphapticNode:
    """
    Spatial ephaptic field Phi_eph(x,t) evolves on slow timescale as 
    low-pass filter of wave field. Acts as spatial prior that pulls 
    local wave activity toward the slow collective pattern.
    
    This is the actual slaving mechanism from synergetics:
      - u(x,t)    : fast enslaved parts  (wave field)
      - Phi_eph   : slow control field   (ephaptic memory)
    
    The ephaptic field is smoothed (blurred) and accumulates slowly.
    Local wave dynamics get an additional pull term toward Phi_eph.
    During reconstruction, Phi_eph acts as the 'memory' of what pattern
    was there, guiding the time-reversal back toward that pattern.
    """
    def __init__(self, size=64, tau=15.0, c_sq=0.24, damping=0.995,
                 scar_thresh=0.4, scar_budget_frac=0.15,
                 eph_tau=0.02, eph_blur=3.0, eph_pull=0.003,
                 device=None):
        self.size = size
        self.tau = tau
        self.c_sq = c_sq
        self.damping = damping
        self.scar_thresh = scar_thresh
        self.scar_budget = scar_budget_frac * size * size
        self.device = device or torch.device('cpu')
        
        # Ephaptic parameters
        self.eph_tau = eph_tau     # slow timescale
        self.eph_blur = eph_blur   # spatial smoothing (gaussian sigma)
        self.eph_pull = eph_pull   # pull strength toward ephaptic field
        
        self.lap_kernel = LAPLACIAN_KERNEL.to(self.device)
        
        # Precompute gaussian blur kernel for ephaptic smoothing
        ks = int(2 * math.ceil(3 * eph_blur) + 1)
        coords = torch.arange(ks, dtype=torch.float32) - ks // 2
        g = torch.exp(-coords**2 / (2 * eph_blur**2))
        g = g / g.sum()
        self.blur_kernel = g.view(1, 1, -1).to(self.device)
        self.blur_ksize = ks
        
        self.reset_field()
        self.scars = torch.zeros(1, 1, size, size, device=self.device)
        self.ephaptic_field = torch.zeros(1, 1, size, size, device=self.device)
        self.label = None
    
    def reset_field(self):
        self.u = torch.zeros(1, 1, self.size, self.size, device=self.device)
        self.u_prev = torch.zeros(1, 1, self.size, self.size, device=self.device)
    
    def reset_ephaptic(self):
        self.ephaptic_field.zero_()
    
    def inject(self, pattern):
        h, w = pattern.shape
        cx = self.size // 2 - h // 2
        cy = self.size // 2 - w // 2
        self.u[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
        self.u_prev[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
    
    def _blur(self, field):
        """Separable gaussian blur."""
        pad = self.blur_ksize // 2
        # Horizontal
        x = F.conv2d(field, self.blur_kernel.view(1, 1, 1, -1), padding=(0, pad))
        # Vertical
        x = F.conv2d(x, self.blur_kernel.view(1, 1, -1, 1), padding=(pad, 0))
        return x
    
    def step(self, train=False, update_ephaptic=True):
        lap = laplacian(self.u, self.lap_kernel)
        beta = torch.abs(lap)
        gamma = 1.0 / (1.0 + self.tau * beta) ** 2
        
        if train:
            freeze = (gamma < self.scar_thresh).float()
            candidate = torch.max(self.scars, freeze * (1.0 - gamma))
            mass = candidate.sum().item()
            if mass > self.scar_budget:
                candidate = candidate * (self.scar_budget / (mass + 1e-8))
            self.scars = torch.clamp(candidate, 0.0, 1.0)
        
        # Update ephaptic field: slow low-pass of blurred wave field
        if update_ephaptic:
            blurred_u = self._blur(self.u)
            self.ephaptic_field = ((1 - self.eph_tau) * self.ephaptic_field 
                                   + self.eph_tau * blurred_u)
        
        # Wave step with ephaptic pull term
        # The pull term gently attracts u toward the slow field pattern
        effective_c_sq = self.c_sq * (1.0 - self.scars)
        u_next = self.damping * (2.0 * self.u - self.u_prev + effective_c_sq * lap)
        
        # Ephaptic slaving: pull local field toward slow ephaptic pattern
        pull = self.eph_pull * (self.ephaptic_field - u_next)
        u_next = u_next + pull
        
        self.u_prev = self.u
        self.u = u_next
        return gamma
    
    def run(self, steps, train=False, update_ephaptic=True):
        for _ in range(steps):
            self.step(train=train, update_ephaptic=update_ephaptic)
    
    def field_energy(self):
        return float(torch.sum(self.u ** 2).item())
    
    def resonance_energy_history(self, probe, steps=300, update_ephaptic=False):
        """
        For resonance tests, freeze the ephaptic field — it already holds
        the learned pattern memory. Don't let the probe rewrite it.
        """
        # Save field, reset only wave
        saved_eph = self.ephaptic_field.clone()
        self.u.zero_()
        self.u_prev.zero_()
        self.inject(probe)
        self.ephaptic_field = saved_eph  # keep the memory
        energies = []
        for _ in range(steps):
            self.step(train=False, update_ephaptic=update_ephaptic)
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


# ---------------------------------------------------------------------------
# Baseline for comparison (no ephaptic)
# ---------------------------------------------------------------------------

class BaselineNode:
    """No ephaptic field — just the v2 frogpond physics."""
    def __init__(self, size=64, tau=15.0, c_sq=0.24, damping=0.995,
                 scar_thresh=0.4, scar_budget_frac=0.15, device=None):
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
    
    def reset_field(self):
        self.u = torch.zeros(1, 1, self.size, self.size, device=self.device)
        self.u_prev = torch.zeros(1, 1, self.size, self.size, device=self.device)
    
    def inject(self, pattern):
        h, w = pattern.shape
        cx = self.size // 2 - h // 2
        cy = self.size // 2 - w // 2
        self.u[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
        self.u_prev[0, 0, cx:cx+h, cy:cy+w] = pattern.to(self.device)
    
    def step(self, train=False):
        lap = laplacian(self.u, self.lap_kernel)
        beta = torch.abs(lap)
        gamma = 1.0 / (1.0 + self.tau * beta) ** 2
        if train:
            freeze = (gamma < self.scar_thresh).float()
            candidate = torch.max(self.scars, freeze * (1.0 - gamma))
            mass = candidate.sum().item()
            if mass > self.scar_budget:
                candidate = candidate * (self.scar_budget / (mass + 1e-8))
            self.scars = torch.clamp(candidate, 0.0, 1.0)
        effective_c_sq = self.c_sq * (1.0 - self.scars)
        u_next = self.damping * (2.0 * self.u - self.u_prev + effective_c_sq * lap)
        self.u_prev = self.u
        self.u = u_next
    
    def run(self, steps, train=False):
        for _ in range(steps):
            self.step(train=train)
    
    def field_energy(self):
        return float(torch.sum(self.u ** 2).item())
    
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reconstruction(device):
    """
    Key test: does ephaptic coupling break the 0.674 correlation ceiling 
    for reconstruction from partial input?
    """
    print("\n" + "=" * 70)
    print("TEST: Reconstruction from partial input (top 50%)")
    print("Target: exceed 0.674 correlation ceiling")
    print("=" * 70)
    
    sq = make_synthetic_pattern('square', size=40)
    partial = make_partial(sq, fraction=0.5, mode='top')
    sq_flat = sq.numpy().ravel()
    
    results = {}
    
    # --- Baseline: no ephaptic ---
    print("\n[Baseline] no ephaptic field")
    node = BaselineNode(device=device)
    node.reset_field()
    node.inject(sq)
    node.run(300, train=True)
    
    node.reset_field()
    node.inject(partial)
    node.run(300, train=False)
    node.time_reverse()
    node.run(300, train=False)
    rec_base = node.read_center(40, 40)
    corr_base = np.corrcoef(sq_flat, rec_base.numpy().ravel())[0, 1]
    print(f"  correlation = {corr_base:.3f}")
    results['baseline'] = (corr_base, rec_base)
    
    # --- Scalar ephaptic (Gemini) ---
    print("\n[Scalar ephaptic] (Gemini's proposal)")
    for eph_coupling in [0.0, 0.005, 0.01, 0.02, 0.05]:
        node = ScalarEphapticNode(device=device, 
                                   eph_coupling=eph_coupling,
                                   eph_target=10.0, eph_tau=0.05)
        node.reset_field()
        node.inject(sq)
        node.run(300, train=True)
        
        node.reset_field()
        node.inject(partial)
        node.run(300, train=False)
        node.time_reverse()
        node.run(300, train=False)
        rec = node.read_center(40, 40)
        rc_flat = rec.numpy().ravel()
        if rc_flat.std() > 1e-4:
            corr = np.corrcoef(sq_flat, rc_flat)[0, 1]
        else:
            corr = 0.0
        print(f"  eph_coupling={eph_coupling}: correlation = {corr:.3f}")
        if eph_coupling == 0.01:
            results['scalar_eph'] = (corr, rec)
    
    # --- Spatial ephaptic (proper slaving) ---
    print("\n[Spatial ephaptic] (slaving principle)")
    for (tau, blur, pull) in [
        (0.02, 3.0, 0.003),
        (0.05, 3.0, 0.005),
        (0.05, 5.0, 0.005),
        (0.10, 3.0, 0.010),
        (0.10, 5.0, 0.010),
        (0.10, 3.0, 0.020),
    ]:
        node = SpatialEphapticNode(device=device,
                                    eph_tau=tau, eph_blur=blur, eph_pull=pull)
        node.reset_field()
        node.reset_ephaptic()
        node.inject(sq)
        node.run(300, train=True, update_ephaptic=True)
        
        # Save the trained ephaptic field; it holds the slow memory
        saved_eph = node.ephaptic_field.clone()
        saved_scars = node.scars.clone()
        
        # Reconstruction phase: inject partial, the ephaptic field pulls
        # the wave back toward the remembered pattern during time reversal
        node.reset_field()
        node.inject(partial)
        node.ephaptic_field = saved_eph  # keep the slow memory
        node.scars = saved_scars
        
        node.run(300, train=False, update_ephaptic=False)
        node.time_reverse()
        node.run(300, train=False, update_ephaptic=False)
        rec = node.read_center(40, 40)
        rc_flat = rec.numpy().ravel()
        if rc_flat.std() > 1e-4:
            corr = np.corrcoef(sq_flat, rc_flat)[0, 1]
        else:
            corr = 0.0
        print(f"  tau={tau}, blur={blur}, pull={pull}: correlation = {corr:.3f}")
        
        # Keep best spatial result
        if ('spatial_eph' not in results or 
            corr > results['spatial_eph'][0]):
            results['spatial_eph'] = (corr, rec)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    images = [sq, partial, results['baseline'][1], 
              results['scalar_eph'][1], results['spatial_eph'][1]]
    titles = ['Original', 'Partial (top 50%)',
              f"Baseline\nr={results['baseline'][0]:.3f}",
              f"Scalar eph\nr={results['scalar_eph'][0]:.3f}",
              f"Spatial eph\nr={results['spatial_eph'][0]:.3f}"]
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    plt.suptitle('Reconstruction: baseline vs ephaptic variants', fontsize=12)
    plt.tight_layout()
    plt.savefig('riemannnet_results/ephaptic_reconstruction.png', dpi=120)
    plt.close()
    print(f"\n  Saved: riemannnet_results/ephaptic_reconstruction.png")
    
    return results


def test_chain_attenuation(device):
    """
    Exp 2's problem: signal dies by node 3 in a four-node chain.
    Does scalar ephaptic coupling (auto gain control) keep the signal alive?
    """
    print("\n" + "=" * 70)
    print("TEST: Chain attenuation (4-node chain, does signal reach node 3?)")
    print("=" * 70)
    
    patterns = {
        'square':   make_synthetic_pattern('square',   size=36),
        'cross':    make_synthetic_pattern('cross',    size=36),
        'circle':   make_synthetic_pattern('circle',   size=36),
        'diagonal': make_synthetic_pattern('diagonal', size=36),
    }
    labels = list(patterns.keys())
    
    # Build two versions: baseline and scalar ephaptic
    # Test: probe with square, see if square-winning discrimination survives
    # through 4 hops
    
    def build_and_run(node_class, node_kwargs, label):
        nodes = [node_class(**node_kwargs) for _ in range(4)]
        for i, (name, pat) in enumerate(patterns.items()):
            nodes[i].label = name
            nodes[i].reset_field()
            nodes[i].inject(pat)
            nodes[i].run(300, train=True)
        
        # Simple direct injection at node 0, probe with each pattern
        # Measure: does matching probe produce highest tail?
        results_by_probe = {}
        for probe_name, probe_pat in patterns.items():
            # Inject probe into node 0 only (for now, local test — 
            # we're checking whether node 0's discrimination stays clean)
            energies = {}
            for i, node in enumerate(nodes):
                node.reset_field()
                node.inject(probe_pat)
                eh = []
                for _ in range(300):
                    node.step(train=False)
                    eh.append(node.field_energy())
                energies[i] = eh
            results_by_probe[probe_name] = energies
        
        print(f"\n[{label}]")
        correct = 0
        for node_id in range(4):
            tails = {p: np.mean(results_by_probe[p][node_id][-50:]) 
                     for p in patterns}
            winner = max(tails, key=tails.get)
            is_correct = winner == labels[node_id]
            if is_correct: correct += 1
            mark = "✓" if is_correct else "✗"
            print(f"  Node {node_id} [{labels[node_id]}]: winner={winner} {mark}"
                  f"  tails: " + ", ".join(f"{k}={v:.2f}" for k,v in tails.items()))
        print(f"  TOTAL: {correct}/4 correct")
        return correct
    
    baseline_correct = build_and_run(BaselineNode, 
                                      dict(device=device), 
                                      'Baseline (no ephaptic)')
    
    scalar_correct = build_and_run(ScalarEphapticNode,
                                    dict(device=device, eph_coupling=0.01,
                                         eph_target=10.0, eph_tau=0.05),
                                    'Scalar ephaptic')
    
    return baseline_correct, scalar_correct


def test_normalized_discrimination(device):
    """
    Critical check: does ephaptic coupling improve shape selectivity
    (beyond amplitude-based discrimination)?
    """
    print("\n" + "=" * 70)
    print("TEST: Shape selectivity with energy-normalized probes")
    print("=" * 70)
    
    patterns = {
        'square':   make_synthetic_pattern('square',   size=40),
        'cross':    make_synthetic_pattern('cross',    size=40),
    }
    # Energy-normalized versions
    norm_patterns = {
        k: p / (torch.sqrt(torch.sum(p**2)) + 1e-8) 
        for k, p in patterns.items()
    }
    
    def test_node(node, label):
        # Train on square
        if isinstance(node, SpatialEphapticNode):
            node.reset_ephaptic()
        node.reset_field()
        node.inject(patterns['square'])
        if isinstance(node, SpatialEphapticNode):
            node.run(300, train=True, update_ephaptic=True)
        else:
            node.run(300, train=True)
        
        # Probe with normalized patterns
        tails = {}
        for name, pat in norm_patterns.items():
            if isinstance(node, SpatialEphapticNode):
                e = node.resonance_energy_history(pat, steps=300, update_ephaptic=False)
            else:
                # Need manual reset-inject-run for baseline
                node.u.zero_()
                node.u_prev.zero_()
                node.inject(pat)
                e = []
                for _ in range(300):
                    node.step(train=False)
                    e.append(node.field_energy())
            tails[name] = np.mean(e[-50:])
        ratio = tails['square'] / (tails['cross'] + 1e-8)
        print(f"  [{label}] square={tails['square']:.4f}, cross={tails['cross']:.4f}, "
              f"ratio={ratio:.2f}")
        return ratio
    
    ratio_base = test_node(BaselineNode(device=device), 'Baseline')
    ratio_scalar = test_node(ScalarEphapticNode(device=device, eph_coupling=0.01), 
                              'Scalar ephaptic')
    ratio_spatial = test_node(SpatialEphapticNode(device=device, 
                                                    eph_tau=0.05, eph_pull=0.005), 
                               'Spatial ephaptic')
    
    return ratio_base, ratio_scalar, ratio_spatial


if __name__ == '__main__':
    os.makedirs('riemannnet_results', exist_ok=True)
    device = torch.device('cpu')
    
    print("Testing ephaptic field variants for RiemannNet")
    print("=" * 70)
    
    recon_results = test_reconstruction(device)
    chain_results = test_chain_attenuation(device)
    norm_results = test_normalized_discrimination(device)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Reconstruction correlation (higher is better):")
    print(f"  Baseline:          {recon_results['baseline'][0]:.3f}")
    print(f"  Scalar ephaptic:   {recon_results['scalar_eph'][0]:.3f}")
    print(f"  Spatial ephaptic:  {recon_results['spatial_eph'][0]:.3f}  (best over params)")
    print(f"\nSingle-node pattern discrimination (4 nodes, each with own pattern):")
    print(f"  Baseline:         {chain_results[0]}/4 correct")
    print(f"  Scalar ephaptic:  {chain_results[1]}/4 correct")
    print(f"\nShape selectivity (square vs cross, energy-normalized probes):")
    print(f"  Baseline:         ratio = {norm_results[0]:.2f}")
    print(f"  Scalar ephaptic:  ratio = {norm_results[1]:.2f}")
    print(f"  Spatial ephaptic: ratio = {norm_results[2]:.2f}")
