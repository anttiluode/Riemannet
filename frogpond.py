import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RiemannResonanceEngine(torch.nn.Module):
    def __init__(self, size=128, tau=15.0):
        super().__init__()
        self.size = size
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The Physics Engine (Laplacian)
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', kernel.to(self.device))
        
        # Permanent Memory (The Frozen Riemann Scars)
        self.register_buffer('frozen_scars', torch.zeros(1, 1, size, size, device=self.device))

    def melt_and_freeze(self, pattern, steps=300):
        """Phase 1: Inject a pattern and let it freeze into the geometry."""
        u = pattern.clone()
        u_prev = pattern.clone()
        for _ in range(steps):
            lap = F.conv2d(u, self.laplacian, padding=1)
            beta = torch.abs(lap)
            gamma = 1.0 / (1.0 + self.tau * beta)**2
            
            # Update Scars: Where Gamma is low, the Scar grows
            # These are the 'Frozen Cores' that won't move during rewind
            freeze = (gamma < 0.4).float()
            self.frozen_scars = torch.max(self.frozen_scars, freeze * (1.0 - gamma))
            
            # Standard Wave Propagation (Damped)
            u_next = 0.98 * (2 * u - u_prev + 0.24 * lap)
            u_prev, u = u, u_next
        return u

    def probe(self, signal, steps=400):
        """Phase 2: Ping the frozen scars with a new signal and listen for the 'Ringing'."""
        u = signal.clone()
        u_prev = signal.clone()
        energies = []
        
        for _ in range(steps):
            lap = F.conv2d(u, self.laplacian, padding=1)
            
            # The Riemann Bumper: Waves cannot enter the frozen scars
            # The speed of time (c_sq) is effectively ZERO inside a scar
            effective_c_sq = 0.24 * (1.0 - self.frozen_scars)
            
            u_next = 0.995 * (2 * u - u_prev + effective_c_sq * lap)
            u_prev, u = u, u_next
            
            # Measure the 'Ringing' (Total energy in the field)
            energies.append(torch.sum(u**2).item())
            
        return u, energies

# --- THE RUN ---
def run():
    engine = RiemannResonanceEngine(size=128).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. TEACH: Inject a Square
    pattern = torch.zeros(1, 1, 128, 128).to(engine.device)
    pattern[0, 0, 40:88, 40:88] = 1.0 # The "A" shape
    print("Writing Square into the Riemann field...")
    engine.melt_and_freeze(pattern)
    
    # 2. PROBE A: Ping with a Square (Matching)
    print("Probing with a matching Square...")
    res_match, energy_match = engine.probe(pattern)
    
    # 3. PROBE B: Ping with a Cross (Non-matching)
    print("Probing with a non-matching Cross...")
    cross = torch.zeros(1, 1, 128, 128).to(engine.device)
    cross[0, 0, 60:68, 20:108] = 1.0
    cross[0, 0, 20:108, 60:68] = 1.0
    res_miss, energy_miss = engine.probe(cross)

    # VISUALIZE THE "RINGING"
    plt.figure(figsize=(12, 4))
    plt.plot(energy_match, label="Resonance (Match)", color='green')
    plt.plot(energy_miss, label="Scattering (Mismatch)", color='red')
    plt.title("The 'Ringing' of the Frozen Nodes")
    plt.xlabel("Time Ticks")
    plt.ylabel("Field Energy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()