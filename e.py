import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class HolographicTimeCrystalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Hologram: 2D Face Reconstruction")
        self.root.geometry("1000x600")
        self.root.configure(bg="#111")

        # --- PHYSICS PARAMETERS ---
        self.grid_size = 150  # Made the pond a bit bigger
        self.c_sq = 0.24      # Perfect stability wave speed
        self.steps_taken = 0
        self.max_steps = 400
        self.is_running = False
        self.direction = 1
        
        self.img_dim = 60     # Big enough to recognize a face
        self.offset = (self.grid_size - self.img_dim) // 2

        # The State Matrices
        self.u = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self.u_prev = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        self.setup_ui()

    def setup_ui(self):
        left_frame = tk.Frame(self.root, bg="#222", width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        btn_style = {"bg": "#444", "fg": "white", "font": ("Consolas", 11), "relief": "flat"}
        tk.Button(left_frame, text="1. Load Face/Image", command=self.load_image, **btn_style).pack(pady=10, fill=tk.X)
        tk.Button(left_frame, text="2. Melt (Forward)", command=self.start_forward, bg="#005500", fg="white", font=("Consolas", 11, "bold")).pack(pady=10, fill=tk.X)
        tk.Button(left_frame, text="3. Crystallize (Reverse)", command=self.start_reverse, bg="#550000", fg="white", font=("Consolas", 11, "bold")).pack(pady=10, fill=tk.X)
        
        self.lbl_status = tk.Label(left_frame, text="Waiting...", bg="#222", fg="#0f0", font=("Consolas", 10))
        self.lbl_status.pack(pady=20)

        right_frame = tk.Frame(self.root, bg="#111")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.lbl_crystal = tk.Label(right_frame, bg="black")
        self.lbl_crystal.pack(expand=True)

        bottom_frame = tk.Frame(right_frame, bg="#111")
        bottom_frame.pack(side=tk.BOTTOM, pady=10)

        self.lbl_input = tk.Label(bottom_frame, bg="black", width=15, height=7)
        self.lbl_input.grid(row=1, column=0, padx=20)
        self.lbl_output = tk.Label(bottom_frame, bg="black", width=15, height=7)
        self.lbl_output.grid(row=1, column=1, padx=20)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        
        # Open image and handle transparency correctly by giving it a black background
        img = Image.open(path).convert("RGBA")
        background = Image.new("RGBA", img.size, (0,0,0,255))
        img_with_bg = Image.alpha_composite(background, img)
        img_gray = img_with_bg.convert('L').resize((self.img_dim, self.img_dim))
        
        # Normalize to 0.0 - 1.0
        img_np = np.array(img_gray, dtype=np.float64) / 255.0
        
        # Show on UI
        img_tk = ImageTk.PhotoImage(img_gray.resize((100, 100), Image.NEAREST))
        self.lbl_input.config(image=img_tk); self.lbl_input.image = img_tk

        # Reset Universe
        self.u.fill(0)
        self.u_prev.fill(0)
        self.steps_taken = 0

        # INJECT IMAGE DIRECTLY INTO THE CENTER OF THE GRID
        o = self.offset
        d = self.img_dim
        self.u[o:o+d, o:o+d] = img_np
        self.u_prev[o:o+d, o:o+d] = img_np

        self.update_crystal_view()
        self.lbl_status.config(text="Image injected at center.\nReady to melt.")

    def laplacian(self, grid):
        # Perfect energy conservation (Torus topology)
        return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)

    def physics_step(self):
        # 2nd Order Wave Equation
        lap = self.laplacian(self.u)
        u_next = 2.0 * self.u - self.u_prev + self.c_sq * lap
        
        self.u_prev = self.u
        self.u = u_next
        self.steps_taken += self.direction

    def run_simulation(self):
        if not self.is_running: return

        for _ in range(4):
            self.physics_step()

        self.update_crystal_view()

        if self.direction == 1 and self.steps_taken >= self.max_steps:
            self.is_running = False
            self.lbl_status.config(text=f"Maximum Entropy Reached.\nTime = {self.steps_taken}")
        elif self.direction == -1 and self.steps_taken <= 0:
            self.is_running = False
            self.lbl_status.config(text=f"Time Reversed.\nReading center.")
            self.read_center()

        if self.is_running:
            self.root.after(16, self.run_simulation)

    def start_forward(self):
        if self.steps_taken >= self.max_steps: return
        self.direction = 1
        self.is_running = True
        self.lbl_status.config(text="Melting into noise...")
        self.run_simulation()

    def start_reverse(self):
        if self.steps_taken <= 0: return
        # Time Reversal Trick
        self.u, self.u_prev = self.u_prev, self.u
        
        self.direction = -1
        self.is_running = True
        self.lbl_status.config(text="Reconstructing face...")
        self.run_simulation()

    def read_center(self):
        # Extract the center 60x60 patch
        o = self.offset
        d = self.img_dim
        recovered_img = self.u[o:o+d, o:o+d].copy()
        
        # AUTO-CONTRAST FIX: Scale the highest value back to white (1.0)
        max_val = np.max(recovered_img)
        if max_val > 0.001:
            recovered_img = recovered_img / max_val
            
        recovered_img = np.clip(recovered_img, 0, 1)
        img_tk = ImageTk.PhotoImage(Image.fromarray((recovered_img * 255).astype(np.uint8)).resize((100, 100), Image.NEAREST))
        self.lbl_output.config(image=img_tk); self.lbl_output.image = img_tk

    def update_crystal_view(self):
        # View auto-scaling
        field = self.u
        m = np.max(np.abs(field)) + 1e-5
        vis = (field / m + 1.0) / 2.0
        vis = np.clip(vis, 0, 1)
        
        img_pil = Image.fromarray((vis * 255).astype(np.uint8), mode='L').resize((450, 450), Image.NEAREST)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.lbl_crystal.config(image=img_tk)
        self.lbl_crystal.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = HolographicTimeCrystalApp(root)
    root.mainloop()