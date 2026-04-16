import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class HolographicTimeStackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phase Hologram: The Time Stack (FILO Memory)")
        self.root.geometry("1000x650")
        self.root.configure(bg="#111")

        # --- PHYSICS PARAMETERS ---
        self.grid_size = 150
        self.c_sq = 0.24      # Perfect stability wave speed
        self.steps_taken = 0
        self.is_running = False
        self.direction = 1
        
        self.img_dim = 60
        self.offset = (self.grid_size - self.img_dim) // 2

        # The State Matrices
        self.u = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self.u_prev = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)

        self.img1_np = None
        self.img2_np = None

        self.setup_ui()

    def setup_ui(self):
        left_frame = tk.Frame(self.root, bg="#222", width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        btn_style = {"bg": "#444", "fg": "white", "font": ("Consolas", 11), "relief": "flat"}
        
        tk.Button(left_frame, text="1. Load Image A (Time 0)", command=lambda: self.load_image(1), **btn_style).pack(pady=10, fill=tk.X)
        tk.Button(left_frame, text="2. Melt Image A (150 steps)", command=lambda: self.start_forward(150), bg="#005500", fg="white", font=("Consolas", 10, "bold")).pack(pady=5, fill=tk.X)
        
        tk.Button(left_frame, text="3. Load Image B (Time 150)", command=lambda: self.load_image(2), **btn_style).pack(pady=(20, 10), fill=tk.X)
        tk.Button(left_frame, text="4. Drop & Melt B (150 steps)", command=self.drop_image_2, bg="#005500", fg="white", font=("Consolas", 10, "bold")).pack(pady=5, fill=tk.X)

        tk.Button(left_frame, text="5. REVERSE TIME", command=self.start_reverse, bg="#550000", fg="white", font=("Consolas", 12, "bold")).pack(pady=(30, 10), fill=tk.X)
        
        self.lbl_status = tk.Label(left_frame, text="Time: 0", bg="#222", fg="#0f0", font=("Consolas", 12))
        self.lbl_status.pack(pady=20)

        right_frame = tk.Frame(self.root, bg="#111")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.lbl_crystal = tk.Label(right_frame, bg="black")
        self.lbl_crystal.pack(expand=True)

        bottom_frame = tk.Frame(right_frame, bg="#111")
        bottom_frame.pack(side=tk.BOTTOM, pady=10)

        tk.Label(bottom_frame, text="Image A", bg="#111", fg="white").grid(row=0, column=0, padx=20)
        tk.Label(bottom_frame, text="Image B", bg="#111", fg="white").grid(row=0, column=1, padx=20)
        tk.Label(bottom_frame, text="Live Center Readout", bg="#111", fg="#0f0").grid(row=0, column=2, padx=20)
        
        self.lbl_input1 = tk.Label(bottom_frame, bg="black", width=15, height=7)
        self.lbl_input1.grid(row=1, column=0, padx=20)
        
        self.lbl_input2 = tk.Label(bottom_frame, bg="black", width=15, height=7)
        self.lbl_input2.grid(row=1, column=1, padx=20)

        self.lbl_output = tk.Label(bottom_frame, bg="black", width=15, height=7)
        self.lbl_output.grid(row=1, column=2, padx=20)

    def load_image(self, slot):
        path = filedialog.askopenfilename()
        if not path: return
        
        img = Image.open(path).convert("RGBA")
        background = Image.new("RGBA", img.size, (0,0,0,255))
        img_with_bg = Image.alpha_composite(background, img)
        img_gray = img_with_bg.convert('L').resize((self.img_dim, self.img_dim))
        
        img_np = np.array(img_gray, dtype=np.float64) / 255.0
        img_tk = ImageTk.PhotoImage(img_gray.resize((100, 100), Image.NEAREST))
        
        if slot == 1:
            self.img1_np = img_np
            self.lbl_input1.config(image=img_tk); self.lbl_input1.image = img_tk
            
            # Reset and Inject Image 1
            self.u.fill(0)
            self.u_prev.fill(0)
            self.steps_taken = 0
            o, d = self.offset, self.img_dim
            self.u[o:o+d, o:o+d] = self.img1_np
            self.u_prev[o:o+d, o:o+d] = self.img1_np
            
            self.update_crystal_view()
            self.lbl_status.config(text=f"Time: {self.steps_taken}\nImage A Ready.")
        else:
            self.img2_np = img_np
            self.lbl_input2.config(image=img_tk); self.lbl_input2.image = img_tk
            self.lbl_status.config(text=f"Time: {self.steps_taken}\nImage B Queued.")

    def drop_image_2(self):
        if self.img2_np is None: return
        o, d = self.offset, self.img_dim
        # ADD image 2 to the existing chaotic wave field (Superposition)
        self.u[o:o+d, o:o+d] += self.img2_np
        self.u_prev[o:o+d, o:o+d] += self.img2_np
        
        self.start_forward(150)

    def laplacian(self, grid):
        return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
                np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)

    def physics_step(self):
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
        self.read_center() # Read the center continuously

        self.lbl_status.config(text=f"Time: {self.steps_taken}")

        # Stop conditions
        if self.direction == 1 and self.steps_taken >= self.target_steps:
            self.is_running = False
        elif self.direction == -1 and self.steps_taken <= 0:
            self.is_running = False
            self.lbl_status.config(text="Time 0 Reached.")

        if self.is_running:
            self.root.after(16, self.run_simulation)

    def start_forward(self, steps):
        self.direction = 1
        self.target_steps = self.steps_taken + steps
        self.is_running = True
        self.run_simulation()

    def start_reverse(self):
        if self.steps_taken <= 0: return
        # Time Reversal Trick
        self.u, self.u_prev = self.u_prev, self.u
        
        self.direction = -1
        self.is_running = True
        self.run_simulation()

    def read_center(self):
        o, d = self.offset, self.img_dim
        recovered_img = self.u[o:o+d, o:o+d].copy()
        
        max_val = np.max(np.abs(recovered_img))
        if max_val > 0.001:
            recovered_img = recovered_img / max_val
            
        recovered_img = np.clip((recovered_img + 1.0)/2.0, 0, 1) if self.direction == 1 else np.clip(recovered_img, 0, 1)
        
        img_tk = ImageTk.PhotoImage(Image.fromarray((recovered_img * 255).astype(np.uint8)).resize((100, 100), Image.NEAREST))
        self.lbl_output.config(image=img_tk); self.lbl_output.image = img_tk

    def update_crystal_view(self):
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
    app = HolographicTimeStackApp(root)
    root.mainloop()